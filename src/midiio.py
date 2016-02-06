from __future__ import division
import midi

class MidiIO:
    '''
    This class forms an abstraction around the python-midi module to process
    and interact with the necessary deep belief network data formats.
    '''

    def __init__(self, path):
        '''
        Initialize the MIDI reader/writer.

        PARAMETERS:
        path (string) path to read/write the MIDI file
        '''
        self._path = path

    def parse_midi(self, time_offset=0.025):
        '''
        Parse the binary MIDI file (@self._path) and convert to list of notes with
        pitch, onset, and offset times in seconds.
            Note: only uses the first track in the MIDI file.

        PARAMETERS:
        time_offset (float): amount of time (s) to delay MIDI note events. For example,
            GuitarPro adds 0.025 seconds of silence to the beginning of the track

        RETURNS:
        notes (list): list of note events. Each element is a dict of the form
            {pitch, onset_tick, onset_s, offset_tick, offset_s}
        '''

        pattern = midi.read_midifile(self._path)
        pattern.make_ticks_abs()
        ppq = pattern.resolution    # pulses per quarter note (ticks per beat)
        track = pattern[0]

        # translate MIDI messages to list of notes: MIDI pitch, onset, offset (seconds)
        # important: must keep track of tempo messages
        notes = []
        tempo = 120     # bpm (beats per minute)
        spqn = 0.5   # spqn (seconds per quarter note)
        for event in track:
            if type(event) is midi.events.SetTempoEvent:
                tempo = event.get_bpm()
                spqn = float(event.get_mpqn())/1e6  # convert ms per quarter note to s per quarter note
            elif type(event) is midi.events.NoteOnEvent:
                note = {
                    'pitch': event.get_pitch(),
                    'onset_tick': event.tick,
                    'onset_s': float(event.tick * (spqn / ppq) + time_offset),
                    'offset_tick': None,
                    'offset_s': None
                }
                notes.append(note)
            elif type(event) is midi.events.NoteOffEvent:
                offset_pitch = event.get_pitch()
                # backwards lookup note onset with same pitch to append data
                for n in reversed(notes):
                    if n['pitch'] == offset_pitch:
                        n['offset_tick'] = event.tick
                        n['offset_s'] = float(event.tick * (spqn / ppq) + time_offset)
                        break
            elif type(event) is midi.events.EndOfTrackEvent:
                # backwards lookup notes who haven't been given offset times
                for n in reversed(notes):
                    if n['offset_tick'] is None:
                        n['offset_tick'] = event.tick
                        n['offset_s'] = float(event.tick * (spqn / ppq) + time_offset)
                    else:
                        break

        # prune notes that weren't given an offset time
        # (this will occur in invalid MIDI where chords contain identical notes)
        for i in range(len(notes)-1,-1,-1):
            if notes[i]['offset_tick'] is None or notes[i]['offset_s'] is None:
                del notes[i]

        return notes

    def write_midi(self, note_events, time_offset=0.025):
        '''
        Given a sequence of note events with properties {pitch, onset_s, offset_s},
        calculate the MIDI ticks for each note and write these to a MIDI file.

        PARAMETERS:
        note_events (list): list of note events
        time_offset (float): amount of time (s) to delay MIDI note events. For example,
            GuitarPro adds 0.025 seconds of silence to the beginning of the track
        '''

        tempo = 120 # beats per minute
        ppq = 220   # pulses per quarter note (ticks per beat)
        v = 60      # note velocity
        pattern = midi.Pattern(resolution=ppq, tick_relative=False) # work in absolute tick mode

        def time_to_tick(time):
            '''
            Helper function that converts time in seconds to absolute MIDI ticks
            '''
            return int((tempo/60) * ppq * time)

        midi_events = []
        midi_events.append(midi.ProgramChangeEvent(tick=0, channel=0, data=[24]))
        midi_events.append(midi.SetTempoEvent(tick=0, bpm=tempo))
        for n in note_events:
            n['onset_tick'] = time_to_tick(n['onset_s'] + time_offset)
            midi_events.append(midi.NoteOnEvent(tick=n['onset_tick'], velocity=v, pitch=n['pitch']))
            n['offset_tick'] = time_to_tick(n['offset_s'] + time_offset)
            midi_events.append(midi.NoteOffEvent(tick=n['offset_tick'], velocity=v, pitch=n['pitch']))
        midi_events.append(midi.EndOfTrackEvent(tick=midi_events[-1].tick+1))

        midi_events = sorted(midi_events, key=lambda x: x.tick)

        track = midi.Track(events=midi_events, tick_relative=False)
        pattern.append(track)
        pattern.make_ticks_rel()    # back to relative delta ticks

        midi.write_midifile(self._path, pattern)

if __name__ == "__main__":
    # the following code is purely for testing purposes
    mio = MidiIO('/home/gburlet/University/MSc/Thesis/deepstarguitar/data/mid/delilah.mid')
    notes = mio.parse_midi()

    mio = MidiIO('/home/gburlet/University/MSc/Thesis/deepstarguitar/data/midout/delilah.mid')
    mio.write_midi(notes)

    #import pickle
    #with open('/home/gburlet/University/MSc/Thesis/deepstarguitar/data/notes.pkl', 'rb') as pkl:
        #note_events = pickle.load(pkl)

    #print note_events[0]
    #mio.write_midi(note_events[0])