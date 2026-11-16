"""Lightweight linking: placement, letterboxing, latch, and missing observations."""

import unittest
from dataclasses import replace

import cv2
import numpy as np

from astra_demo.linked_views import LinkedSideView, camera_point
from astra_demo.operator_panel import OperatorPanel
from astra_demo.tracking import HandObservation


def hand(x=200.,y=280.):
    palm=np.array([x,y])
    return HandObservation(palm,np.eye(2),80.,palm-[0,40],.7,.05,
        palm+[[-25,30],[-40,-30],[0,-40],[25,-30],[40,0]])


class LinkedViewTests(unittest.TestCase):
    def test_no_ball_orbit_when_side_palm_angle_and_scale_jump(self):
        view=LinkedSideView();view.set_frame_size((640,480));view.place([200,240])
        frame=np.zeros((480,640,3),np.uint8)
        obs=hand()
        view.update(frame,obs,True,1.)
        positions=[]
        for i in range(1,61):
            angle=(-1)**i*1.4
            basis=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            shaky=replace(obs,basis=basis,scale=40.+(i%4)*30.,
                          palm=obs.palm+[2*(-1)**i,3*(-1)**i],pinch_xy=obs.pinch_xy+[35*(-1)**i,30])
            view.update(frame,shaky,True,1.+i/15)
            positions.append(view.prop.pos_xy)
        self.assertLess(np.abs(np.asarray(positions)-[200,250]).max(),2.)
        for i in range(1,21): view.update(frame,hand(200,280-i*5),True,5.+i/15)
        self.assertLess(abs(view.prop.pos_xy[1]-150),16.)
        for i in range(1,21): view.update(frame,hand(200,180+i*5),True,7.+i/15)
        self.assertLess(abs(view.prop.pos_xy[1]-250),16.)

    def test_transient_side_detection_jump_freezes_then_recovers(self):
        view=LinkedSideView();view.set_frame_size((640,480));view.place([200,240])
        frame=np.zeros((480,640,3),np.uint8)
        view.update(frame,hand(),True,1.)
        view.update(frame,hand(260,340),True,1.1)
        self.assertEqual(view.prop.pos_xy,(200.,250.))
        self.assertEqual(view.prop.state.value,'HELD')
        view.update(frame,hand(),True,1.2)
        np.testing.assert_allclose(view.prop.pos_xy,[200.,250.])

    def test_placement_reset_and_latched_grasp(self):
        view=LinkedSideView()
        view.set_frame_size((640,480))
        frame=np.zeros((480,640,3),np.uint8)
        self.assertFalse(view.ready)
        view.place([200,240])
        view.update(frame,hand(300,280),False,1.)
        self.assertEqual(view.prop.pos_xy,(200.,240.))
        view.update(frame,hand(),True,1.1)
        for i in range(1,7): view.update(frame,hand(200,280-i*12),True,1.1+i*.05)
        self.assertLess(view.prop.pos_xy[1],200)
        last=view.prop.pos_xy
        view.update(None,None,True,2.)
        self.assertEqual(view.prop.pos_xy,last)
        self.assertEqual(view.prop.state.value,'HELD')
        self.assertIsNone(view.record(None)['side_palm_y_px'])
        self.assertEqual(view.record(None)['side_position_valid'],0)
        view.update(frame,hand(400,360),False,2.1)
        self.assertEqual(view.prop.pos_xy,last)
        self.assertEqual(view.prop.state.value,'IDLE')
        view.reset()
        self.assertEqual(view.prop.pos_xy,(200.,240.))
        with self.assertRaises(ValueError): view.place([20,20],locked=True)
        with self.assertRaises(ValueError): view.place([640,200])

    def test_window_and_console_clicks_ignore_padding_after_resize(self):
        self.assertEqual(camera_point(300,400,(100,200,2,2),(640,480)),(100.,100.))
        self.assertIsNone(camera_point(99,300,(100,200,2,2),(640,480)))
        panel=OperatorPanel()
        panel.linked_views=True
        panel.render(side=np.zeros((480,640,3),np.uint8))
        dx,dy,sx,sy=panel._side_camera_transform
        panel.on_mouse(cv2.EVENT_LBUTTONDOWN,837,200,0,None)
        self.assertIsNone(panel.side_place_request)
        panel._display_transform=(10,20,2,2)
        panel._on_window_mouse(cv2.EVENT_LBUTTONDOWN,10+2*(dx+320*sx),20+2*(dy+240*sy),0,None)
        np.testing.assert_allclose(panel.side_place_request,[320,240])

    def test_resolution_change_requires_new_placement_without_releasing(self):
        view=LinkedSideView()
        view.set_frame_size((640,480));view.place([200,240])
        view.update(np.zeros((480,640,3),np.uint8),hand(),True,1.)
        self.assertTrue(view.set_frame_size((1280,960)))
        self.assertFalse(view.ready)
        view.update(None,None,True,1.2)
        self.assertEqual(view.prop.state.value,'HELD')
        np.testing.assert_allclose(view.prop.pos_xy,[400,500])

    def test_inset_is_captured_once_without_affecting_raw_hand_or_release(self):
        view=LinkedSideView();view.set_frame_size((640,480));view.place([200,240])
        frame=np.zeros((480,640,3),np.uint8)
        obs=hand()
        original=obs.pinch_xy.copy()
        view.update(frame,obs,False,1.)
        np.testing.assert_allclose(view.prop.pos_xy,[200,240])
        view.update(frame,obs,True,1.1)
        np.testing.assert_allclose(view.prop.pos_xy,[200,250])
        np.testing.assert_allclose(obs.pinch_xy,original)
        changed=replace(obs,pinch_xy=obs.pinch_xy+[80,40],scale=150.,basis=-obs.basis)
        view.update(frame,changed,True,1.2)
        np.testing.assert_allclose(view.prop.pos_xy,[200,250])
        view.update(frame,changed,False,1.3)
        np.testing.assert_allclose(view.prop.pos_xy,[200,250])
        view.reset()
        np.testing.assert_allclose(view.prop.pos_xy,[200,240])


if __name__=='__main__': unittest.main()
