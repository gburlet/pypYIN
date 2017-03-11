# -*- coding: utf-8 -*-

'''
 * Copyright (C) 2015  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of pypYIN
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 * If you have any problem about this algorithm, I suggest you to contact: Matthias Mauch
 * m.mauch@qmul.ac.uk who is the original C++ version author of this algorithm
 *
 * If you want to refer this code, please consider this article:
 *
 * M. Mauch and S. Dixon,
 * “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions”,
 * in Proceedings of the IEEE International Conference on Acoustics,
 * Speech, and Signal Processing (ICASSP 2014), 2014.
 *
 * M. Mauch, C. Cannam, R. Bittner, G. Fazekas, J. Salamon, J. Dai, J. Bello and S. Dixon,
 * “Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency”,
 * in Proceedings of the First International Conference on Technologies for
 * Music Notation and Representation, 2015.
'''

import os
import sys
import time

dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import pYINmain
import librosa
import numpy as np
from midiio import MidiIO

if __name__ == "__main__":

    # initialise
    filename1 = '/media/gburlet/Beartracks/Frettable/data/vocals/all/audio/musetta_-_Ophelia_s_Song_(Vocals).wav'
    sr = 44100
    frameSize = 2048
    hopSize = 256

    t0 = time.time()
    pYinInst = pYINmain.PyinMain()
    pYinInst.initialise(
        channels=1, inputSampleRate=sr, stepSize=hopSize, blockSize=frameSize,
        lowAmp=0.25, onsetSensitivity=0.7, pruneThresh=0.1
    )

    y, _ = librosa.load(filename1, sr=sr, mono=True)
    frames = librosa.util.frame(y, frame_length=frameSize, hop_length=hopSize)
    pYinInst.process(frames)

    # calculate smoothed pitch and mono note
    note_track = pYinInst.getNoteTrack()

    # {pitch, onset_s, offset_s},
    # write midi
    notes = []
    for note in note_track:
        notes.append({
            "pitch": note["midi_number"],
            "onset_s": note["onset_time_s"],
            "offset_s": note["onset_time_s"] + note["length_s"]
        })

    mio = MidiIO('/home/gburlet/Desktop/ophelia_pypYIN.mid')
    mio.write_midi(notes)
    t1 = time.time()
    total_time = t1-t0

    print "Transcribed in %0.2f seconds" % total_time
