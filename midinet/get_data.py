import xml.etree.ElementTree as ET 
import xmldataset
import os 
from os.path import basename, dirname, join, exists, splitext
import ipdb
import numpy as np
import json
from math import floor, ceil

def get_sample(cur_song, cur_dur,n_ratio, dim_pitch, dim_bar, cancion):

    cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
    idx = 1
    sd = 0
    ed = 0
    song_sample=[]
    
    if cancion == 1528:
        print("Total notas shakira: {}".format(len(cur_song)))
    while idx < len(cur_song):
        
        cur_pitch = cur_song[idx]-1
        if cancion == 1528:
            print(cur_pitch)
        ed = int(ed + cur_dur[idx]*n_ratio)
        # print('pitch: {}, sd:{}, ed:{}'.format(cur_pitch, sd, ed))
        if ed <dim_bar:
            cur_bar[0,cur_pitch,sd:ed]=1
            sd = ed
            idx = idx +1
        elif ed >= dim_bar:
            cur_bar[0,cur_pitch,sd:]=1
            song_sample.append(cur_bar)
            cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
            sd = 0
            ed = 0
            # print(cur_bar)
            # print(song_sample)
        # if idx == len(cur_song)-1 and np.sum(cur_bar)!=0:
        #     song_sample.append(cur_bar)
    return song_sample

def build_matrix2(note_list_all_c,dur_list_all_c, c_files):
    data_x = []           
    prev_x = []
    zero_counter = 0
    for i in range(len(note_list_all_c)):
        if i == 26:
            print("This song is: {}".format(c_files[i]))
            song = note_list_all_c[i]
            dur = dur_list_all_c[i]
            song_sample = get_sample(song,dur,4,128,128)
            print(type(song_sample))
            print(song_sample)
            np_sample = np.asarray(song_sample)
            print(np_sample.shape)
            if len(np_sample) == 0:
                zero_counter +=1
            if len(np_sample) != 0:
                np_sample =np_sample[0]
                np_sample = np_sample.reshape(1,1,128,128)

                if np.sum(np_sample) != 0:
                    place = np_sample.shape[3]
                    new=[]
                    for i in range(0,place,16):
                        new.append(np_sample[0][:,:,i:i+16])
                    new = np.asarray(new)  # (2,1,128,128) will become (16,1,128,16)
                    print(np_sample.shape)
                    print(new.shape)
                    new_prev = np.zeros(new.shape,dtype=int)
                    new_prev[1:, :, :, :] = new[0:new.shape[0]-1, :, :, :]            
                    data_x.append(new)
                    prev_x.append(new_prev)  

    data_x = np.vstack(data_x)
    prev_x = np.vstack(prev_x)


    return data_x,prev_x,zero_counter

def build_matrix(note_list_all_c,dur_list_all_c, chord_list_all):
    data_x = []           
    prev_x = []
    chords = []
    zero_counter = 0
    for i in range(len(note_list_all_c)):
        #if i < len(c_files):
            #print("This song is: {}".format(c_files[i]))
        song = note_list_all_c[i]
        dur = dur_list_all_c[i]
        song_sample = get_sample(song,dur,4,128,128, i)
        song_chord = chord_list_all[i]
        #print('Total of chords in song: {}'.format(len(song_chord)))
        #print(type(song_sample))
        #print(song_sample)
        np_sample = np.asarray(song_sample)
        
        if len(song_sample)*8 > len(song_chord):
            #print("Canción: {}, idx: {}".format(list_files[i], i))
            print("Num compases: {}, acordes: {}".format(len(song_sample), len(song_chord)))
            continue
        #if i < len(c_files):
            #print("Shape of np_sample: {}".format(np_sample.shape))
        if len(np_sample) == 0:
            zero_counter +=1
        if len(np_sample) != 0:
            chunk = 0
            for sample in np_sample:
                sample = sample.reshape(1,1,128,128)
                if np.sum(sample) != 0:
                    place = sample.shape[3]
                    new=[]
                    for j in range(0,place,16):
                        new.append(sample[0][:,:,j:j+16])
                    new = np.asarray(new)  # (2,1,128,128) will become (16,1,128,16)
                    
                    new_chord = song_chord[chunk:chunk+8]
                    chunk += 8
                    new_chord = np.asarray(new_chord)

                                      #print(sample.shape)
                    
                    #print(new.shape)
                    new_prev = np.zeros(new.shape,dtype=int)
                    new_prev[1:, :, :, :] = new[0:new.shape[0]-1, :, :, :]            
                    data_x.append(new)
                    prev_x.append(new_prev)
                    chords.append(new_chord)
    '''
    for chord in chords:
        if len(chord) != 13:
            print(chord.shape)'''

    data_x = np.vstack(data_x)
    prev_x = np.vstack(prev_x)
    chords = np.vstack(chords)


    return data_x,prev_x,chords,zero_counter




def check_melody_range(note_list_all,dur_list_all, file_list):
    in_range=0
    note_list_all_c = []
    dur_list_all_c = []
    list_files = []
    for i in range(len(note_list_all)):
        song = note_list_all[i]
        if len(song[1:]) ==0:
            print("error")
            ipdb.set_trace()
        elif min(song[1:])>= 60 and max(song[1:])<= 83:
            in_range +=1
            note_list_all_c.append(song)
            dur_list_all_c.append(dur_list_all[i])
            list_files.append(file_list[i])

    np.save('dur_list_all_c.npy',dur_list_all_c)
    np.save('note_list_all_c.npy',note_list_all_c)

    return in_range,note_list_all_c,dur_list_all_c, list_files

def transform_note(c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list):
    scale = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93]
    transfor_list_C1 = scale[0:7]
    transfor_list_C2 = scale[7:14]
    transfor_list_C3 = scale[14:21]

    transfor_list_D1 = scale[1:8]
    transfor_list_D2 = scale[8:15]
    transfor_list_D3 = scale[15:22]

    transfor_list_E1 = scale[2:9]
    transfor_list_E2 = scale[9:16]
    transfor_list_E3 = scale[16:23]

    transfor_list_F1 = scale[3:10]
    transfor_list_F2 = scale[10:17]
    transfor_list_F3 = scale[17:24]

    transfor_list_G1 = scale[4:11]
    transfor_list_G2 = scale[11:18]
    transfor_list_G3 = scale[18:25]

    transfor_list_A1 = scale[5:12]
    transfor_list_A2 = scale[12:19]
    transfor_list_A3 = scale[19:26]

    transfor_list_B1 = scale[6:13]
    transfor_list_B2 = scale[13:20]
    transfor_list_B3 = scale[20:27]
    fails = 0
    note_c =[]  
    dur_c =[]
    c_files = []
    d_files = []
    e_files = []
    f_files = []
    g_files = []
    a_files = []
    b_files = []
    appends = 0
    print("comienzo keys\nc: {}\nd:{}\ne:{}\nf:{}\ng:{}\na:{}\nb:{}".format(len(c_key_list), len(d_key_list), len(e_key_list), len(f_key_list), len(g_key_list), len(a_key_list), len(b_key_list)))
    for file_ in c_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_C1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_C2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_C3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_c.append(note_list)
            dur_c.append(dur_list)
            c_files.append(file_)
            appends += 1

        except:
            #print('c key but no melody/notes :{}'.format(file_))
            fails += 1

    note_d = []
    dur_d = []
    for file_ in d_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_D1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_D2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_D3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_d.append(note_list)
            dur_d.append(dur_list)
            d_files.append(file_)
            appends += 1

        except:
            #print('d key but no melody/notes :{}'.format(file_))
            fails += 1

    note_e = []
    dur_e = []
    for file_ in e_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_E1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_E2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_E3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_e.append(note_list)
            dur_e.append(dur_list)
            e_files.append(file_)
            appends += 1

        except:
            #print('e key but no melody/notes :{}'.format(file_))
            fails += 1

    note_f = []
    dur_f = []
    for file_ in f_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_F1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_F2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_F3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_f.append(note_list)
            dur_f.append(dur_list)
            f_files.append(file_)
            appends += 1

        except:
            #print('f key but no melody/notes :{}'.format(file_))
            fails += 1


    note_g = []
    dur_g = []
    for file_ in g_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_G1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_G2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_G3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_g.append(note_list)
            dur_g.append(dur_list)
            g_files.append(file_)
            appends += 1

        except:
            #print('g key but no melody/notes :{}'.format(file_))
            fails += 1

    note_a = []
    dur_a = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_A1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_A2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_A3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_a.append(note_list)
            dur_a.append(dur_list)
            a_files.append(file_)
            appends += 1

        except:
            #print('a key but no melody/notes :{}'.format(file_))
            fails += 1


    note_b = []
    dur_b = []
    for file_ in b_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_B1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_B2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_B3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_b.append(note_list)
            dur_b.append(dur_list)
            b_files.append(file_)
            appends += 1

        except:
            #print('b key but no melody/notes :{}'.format(file_))
            fails += 1
   
    #print("han habido {} appends".format(appends))
    #print("acabo keys\nc: {}\nd:{}\ne:{}\nf:{}\ng:{}\na:{}\nb:{}".format(len(note_c), len(note_d), len(note_e), len(note_f), len(note_g), len(note_a), len(note_b)))
    #print(fails)
    note_list_all = note_c + note_d + note_e + note_f + note_g + note_a + note_b
    dur_list_all = dur_c + dur_d + dur_e  + dur_f + dur_g + dur_a  + dur_b
    file_list_all = c_files + d_files + e_files + f_files + g_files + a_files + b_files
    #print("Note list all: {}, dur list all: {}, file list all: {}".format(len(note_list_all), len(dur_list_all), len(file_list_all)))
    return note_list_all,dur_list_all, file_list_all

def get_key(list_of_four_beat):
    key_list =[]
    c_key_list = []
    d_key_list = []
    e_key_list = []
    f_key_list = []
    g_key_list = []
    a_key_list = []
    b_key_list = []
    for file_ in list_of_four_beat:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            key = root.findall('.//key')
            key_list.append(key[0].text)
            if key[0].text == 'C':
                c_key_list.append(file_)
            if key[0].text == 'D':
                d_key_list.append(file_)
            if key[0].text == 'E':
                e_key_list.append(file_) 
            if key[0].text == 'F':
                f_key_list.append(file_)
            if key[0].text == 'G':
                g_key_list.append(file_) 
            if key[0].text == 'A':
                a_key_list.append(file_)  
            if key[0].text == 'B':
                b_key_list.append(file_)                            
        except:
            print('file broken')
    # print('A key: {}'.format(key_list.count('A')))
    # print('B key: {}'.format(key_list.count('B')))
    # print('C key: {}'.format(key_list.count('C')))
    # print('D key: {}'.format(key_list.count('D')))
    # print('E key: {}'.format(key_list.count('E')))
    # print('F key: {}'.format(key_list.count('F')))
    # print('G key: {}'.format(key_list.count('G')))
    return c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list

def beats_(list_):
    list_of_four_beat =[]
    for file_ in list_:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            beats = root.findall('.//beats_in_measure')
            num = beats[0].text
            if num == '4':
                list_of_four_beat.append(file_) 
        except:
            print('cannot open the file')
    return list_of_four_beat

def check_chord_type(list_file):
    list_ = []
    for file_ in list_file:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            check_list = []
            counter = 0
            None_counter = 0
            for item in root.iter(tag='fb'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            for item in root.iter(tag='borrowed'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            #print(check_list)
            #print(counter)
            #print(None_counter)
            if counter == None_counter :
                list_.append(file_)
        except:
            print('cannot open')
    return list_ #, check_list


def multiples_interval(interval):
    # interval tuple of float
    low = ceil(interval[0])
    up = ceil(interval[1])
    multiples = 0
    for i in range(low, up):
        if i%4 == 0:
            multiples += 1
    return multiples

def get_chord(list_file):
    chord_list = []
    for file in list_file:
        song_chord = []
        with open(file) as json_file:
            data_nokey = json.load(json_file)
        chords = data_nokey['tracks']['chord'] # is a list of dicts
        for chord in chords:
            if chord is not None:
                multiples = multiples_interval((chord['event_on'], chord['event_off']))
                if multiples > 0:
                    mn_chord = np.zeros(13)
                    if chord['quality'] == '':
                        mn_chord[chord['root']%12] = 1
                    elif chord['quality'] == 'm':
                        mn_chord[(chord['root'] + 9)%12] = 1
                        mn_chord[-1] = 1
                    for i in range(multiples):
                        song_chord.append(mn_chord)
        chord_list.append(song_chord)
    np.save('list_all_chords.npy', chord_list)
    return chord_list
                    



def get_listfile(dataset_path):

    list_file=[]

    for root, dirs, files in os.walk(dataset_path):    
        for f in files:
            if splitext(f)[0]=='chorus':                
                fp = join(root, f)
                list_file.append(fp)

    return list_file

def get_jsonfiles(dataset_path):

    list_file=[]

    for root, dirs, files in os.walk(dataset_path):    
        for f in files:
            if splitext(f)[0]=='chorus_symbol_nokey':                
                fp = join(root, f)
                list_file.append(fp)

    return list_file

def xml2json(list_file):
    json_files = []
    new_list_file = []
    for file in list_file:
        json_path = '..\\..\\datasets\\event\\' + '\\'.join(file.split('\\')[4:-1]) + '\\chorus_symbol_nokey.json'
        json_files.append(json_path)
    return json_files

def prune_json(list_file):
    new_list_file = []
    for file in list_file:
        json_path = '..\\..\\datasets\\event\\' + '\\'.join(file.split('\\')[4:-1]) + '\\chorus_symbol_nokey.json'
        if os.path.exists(json_path):
            new_list_file.append(file)
    return new_list_file

def data_augmentation(data, prev, chords):
    # data shape: [13987, 1, 128, 16]
    new_data = []
    new_prev = []
    new_chords = []
    for i in range(len(data)):
        print("Canción {} aumentada".format(i))
        try:
            it = np.argwhere(chords[i,:-1] == 1)[0][0] # nos quedamos con el primer 1
            for key in range(6):
                temp = np.zeros((1,128,16))
                temp[0, (key+1):] = data[i, 0, :-(key+1)]
                new_data.append(temp)
                #print("hago new_Data concatenate: ", new_data.shape)

                temp_prev = np.zeros((1,128,16))
                temp_prev[0, (key+1):] = prev[i, 0, :-(key+1)]
                new_prev.append(temp_prev)
                #print("hago new_prev concatenate: ", new_prev.shape)

                temp_chord = np.zeros(13)
                temp_chord[-1] = chords[i, -1]
                it += (key+1)
                temp_chord[it%12] = 1
                new_chords.append(temp_chord)
                if key < 5:
                    temp[0, :-(key+1)] = data[i, 0, (key+1):]
                    new_data.append(temp)

                    temp_prev[0, :-(key+1)] = prev[i, 0, (key+1):]
                    new_prev.append(temp_prev)

                    it -= 2*(key+1)
                    temp_chord = np.zeros(13)
                    temp_chord[-1] = chords[i, -1]
                    temp_chord[it%12] = 1
                    new_chords.append(temp_chord)
                    it += (key+1)
        except:
            continue
    return np.vstack((data, new_data)), np.vstack((prev, new_prev)), np.vstack((chords, new_chords))

def data_augmentation_chords(data):
    new_data = []
    for chord in data:
        it = np.argwhere(chord[:-1] == 1)[0][0] # nos quedamos con el primer 1
        for key in range(6):
            # subimos
            temp = np.zeros(13)
            temp[-1] = chord[-1]
            it += (key+1)
            temp[it%12] = 1
            new_data.append(temp)
            if key < 5:
            # bajamos
                it -= 2*(key+1)
                temp = np.zeros(13)
                temp[-1] = chord[-1]
                temp[it%12] = 1
                new_data.append(temp)
                it += (key+1)
    return np.vstack(new_data)


def main():
    is_get_data = 0
    is_get_matrix = 1
    if is_get_data == 1:
        a = '..\\..\\datasets\\xml\\'
        json_path = '..\\..\\datasets\\event\\'
        list_file = get_listfile(a)
        list_file = prune_json(list_file)
        list_of_four_beat = beats_(list_file)

        #print("Total antes get_key: {}".format(len(list_of_four_beat)))

        c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list = get_key(list_of_four_beat)

        #print("Total desp get_key: {}".format(len(c_key_list) + len(d_key_list) + len(e_key_list) + len(f_key_list) + len(g_key_list) + len(a_key_list) + len(b_key_list)))

        note_list_all,dur_list_all, list_files = transform_note(c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list)
        in_range,note_list_all_c,dur_list_all_c, list_files = check_melody_range(note_list_all,dur_list_all, list_files)

        #print("total de files según list_file: {}, total según melody: {}".format(len(list_files), len(note_list_all_c)))
        
        json_files = xml2json(list_files)
        chords = get_chord(json_files)
        #print('total normal chord: {}'.format(len(list_)))
        print('total in four: {}'.format(len(list_of_four_beat)))
        print('melody in range: {}'.format(len(note_list_all)))
        print('total chords: {}'.format(len(chords)))

    if is_get_matrix == 1:
        note_list_all_c = np.load('note_list_all_c.npy', allow_pickle=True)
        dur_list_all_c = np.load('dur_list_all_c.npy', allow_pickle=True)
        list_all_chord = np.load('list_all_chords.npy', allow_pickle=True)

        data_x, prev_x, chords, zero_counter = build_matrix(note_list_all_c,dur_list_all_c, list_all_chord)#, list_files)
        print('sample shape: {}, prev sample shape: {}'.format(data_x.shape, prev_x.shape))
        print('Chord shape: {}'.format(chords.shape))
        np.save('data_x.npy',data_x)
        np.save('prev_x.npy',prev_x)
        np.save('chords.npy', chords)
        print("hacemos data augmentation")
        data_x, prev_x, chords = data_augmentation(data_x, prev_x, chords)
        print('sample shape: {}, prev sample shape: {}'.format(data_x.shape, prev_x.shape))
        print('Chord shape: {}'.format(chords.shape))
        #chords = build_chords(list_all_chord, data_x)
        np.save('data_x_augmented.npy',data_x)
        np.save('prev_x_augmented.npy',prev_x)
        np.save('chords_augmented.npy', chords)
        '''
        print('final tab num: {}'.format(len(note_list_all_c)))
        print('songs not long enough: {}'.format(zero_counter))
        print('sample shape: {}, prev sample shape: {}'.format(data_x.shape, prev_x.shape))
        print('Chord shape: {}'.format(chords.shape))'''
    
if __name__ == "__main__" :

    main()
