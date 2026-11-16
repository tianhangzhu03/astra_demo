"""One row per host-clock second, derived from the authoritative exact event log."""

from collections import defaultdict
import csv
import json
import math


def write_feedback_seconds(directory, session_id, end_s):
    # Deferred until close so delayed BLE events retain their original time bucket.
    # events.csv is flushed live and is sufficient to regenerate this derived file.
    buckets = defaultdict(list)
    fields = ('t_s', 'trial_id', 'event', 'command_id', 'cue', 'role', 'reason',
              'commanded_voltage_mv', 'commanded_duration_s', 'frequency_hz', 'direction_code', 'commanded_channels_json',
              'repeat_id', 'repeat_index', 'repeat_interval_s')
    with (directory/'events.csv').open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if row['event'].startswith('command_') or row['event'] == 'cue_unavailable':
                timestamp = float(row['t_s'])
                buckets[max(0, math.floor(timestamp))].append({k: row.get(k, '') for k in fields})
                end_s = max(end_s, timestamp)
    with (directory/'feedback_1hz.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, ['session_id', 'window_start_s', 'window_end_s', 'partial_window',
            'sent_count', 'simulated_count', 'sent_cues', 'simulated_cues', 'unavailable_count', 'events_json'])
        writer.writeheader()
        # Include a terminal event exactly on an integer boundary as an event-only
        # partial row, rather than silently dropping its bucket.
        for second in range(max(1, math.ceil(end_s), max(buckets, default=-1)+1)):
            events = sorted(buckets.get(second, []), key=lambda event: float(event['t_s']))
            sent = [e['cue'] for e in events if e['event'] == 'command_sent']
            simulated = [e['cue'] for e in events if e['event'] == 'command_simulated']
            stop = min(second+1, end_s)
            writer.writerow(dict(session_id=session_id, window_start_s=second, window_end_s=stop,
                partial_window=int(stop-second < 1), sent_count=len(sent), simulated_count=len(simulated),
                sent_cues=json.dumps(sent), simulated_cues=json.dumps(simulated),
                unavailable_count=sum(e['event'] == 'cue_unavailable' for e in events),
                events_json=json.dumps(events, separators=(',', ':'))))
