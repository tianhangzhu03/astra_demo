#ifndef QSIB_CEL_DIRECTION_MAP_H
#define QSIB_CEL_DIRECTION_MAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Extension to the CEL_GUI 0x10 command. Existing IDs 0..4 are unchanged. */
enum qsib_cel_direction {
    QSIB_CEL_NORMAL = 0,
    QSIB_CEL_UP = 1,
    QSIB_CEL_DOWN = 2,
    QSIB_CEL_LEFT = 3,
    QSIB_CEL_RIGHT = 4,
    QSIB_CEL_TORSION = 5
};

/* Returns the colleague-specified states in channel order 0,1,2,3.
 * This function does not choose GPIO pins, set voltage, or energize hardware.
 * The caller must retain the existing stop/zero-voltage handling. On false,
 * reject the command via that handler; do not reinterpret an unknown ID.
 */
static inline bool qsib_cel_direction_channels(uint8_t direction, uint8_t states[4])
{
    static const uint8_t patterns[6][4] = {
        {1, 1, 1, 1}, /* 0: Normal */
        {1, 1, 0, 0}, /* 1: Up */
        {0, 0, 1, 1}, /* 2: Down */
        {1, 0, 1, 0}, /* 3: Left */
        {0, 1, 0, 1}, /* 4: Right */
        {1, 0, 0, 1}  /* 5: Torsion, physical CW/CCW polarity unspecified */
    };
    if (states == NULL || direction >= 6) {
        return false;
    }
    for (uint8_t channel = 0; channel < 4; ++channel) {
        states[channel] = patterns[direction][channel];
    }
    return true;
}

#endif /* QSIB_CEL_DIRECTION_MAP_H */
