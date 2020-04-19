# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:55:13 2020

@author: Robert
"""

import pretty_midi
import numpy as np

midi_file_name = 'MIDI1.midi'
fs = 100

abba_midi = pretty_midi.PrettyMIDI(midi_file_name)
piano_midi = abba_midi.instruments[0] 
piano_roll = piano_midi.get_piano_roll(fs=fs)


    
key = 0
dict_note = {}
for i in np.transpose(piano_roll):
    if i.any():
        notes = np.argwhere(i != 0)
        dict_note[key] = list(zip(np.ravel(notes), np.ravel(i[notes])))
    key += 1
    














