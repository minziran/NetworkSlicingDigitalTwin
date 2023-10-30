import csv
import re
import os
import numpy as np
from datetime import datetime


def split_key(distance_list, key, string):
    s_list = re.split(string, key)
    # for i in s_list:
    #     if i != '':
    #         distance_list.append(float(i))
    distance_list.append(float(s_list[-1]))


def IPProcessing(distance_list, key, string):
    s_list = re.split(string, key)
    # print("s_list",s_list)
    num = ''
    for i in s_list:
        if i != '':
            if len(i)==1:
                num+='00'+ i
            elif len(i)==2:
                num+='0'+ i
            else:
                num+=i
    distance_list.append(float(num))
    # print(distance_list)


def FranceDatasetTimeProcessing(distance_list, key, string):
    # print(key)
    s_list = re.split(string, key)
    # print("s_list", s_list)
    temp =[]
    for i in s_list:
        if i != '':
            temp.append(int(i))    
    if "P" in key and temp[3] != 12:
        temp[3] += 12
    # print(temp)
    dtime = datetime(temp[2], temp[1], temp[0], temp[3],temp[4],temp[5])
    distance_list.append(dtime.timestamp())


def UnicaucaDatasetTimeProcessing(distance_list, key, string): #
    s_list = re.split(string, key)
    # print(s_list)
    temp = []
    for i in s_list:
        if i != '':
            temp.append(i)
    #print(temp)
    dtime = datetime(int(temp[2][:4]), int(temp[1]), int(temp[0]), int(temp[2][4:]), int(temp[3]), int(temp[4]))
    #print(dtime)
    time_str = dtime.strftime("%I:%M:%S %p")
    if "A" in time_str:
        tlabel = "Time AM"
    else:
        tlabel = "Time PM"
    #print(dtime)
    distance_list.append(dtime.timestamp())
    return tlabel


# FranceDataset
# 0 Flow ID 1 Src IP 2 Src Port 3 Dst IP 4 Dst Port 5 Protocol 6 Timestamp 7 Flow Duration 8 Tot Fwd Pkts 9 Tot Bwd Pkts 10 TotLen Fwd Pkts
# 11 TotLen Bwd Pkts 12 Fwd Pkt Len Max 13 Fwd Pkt Len Min 14 Fwd Pkt Len Mean 15 Fwd Pkt Len Std 16 Bwd Pkt Len Max 17 Bwd Pkt Len Min 18 Bwd Pkt Len Mean 19 Bwd Pkt Len Std 20 Flow Byts/s
# 21 Flow Pkts/s 22 Flow IAT Mean 23 Flow IAT Std 24 Flow IAT Max 25 Flow IAT Min 26 Fwd IAT Tot 27 Fwd IAT Mean 28 Fwd IAT Std 29 Fwd IAT Max 30 Fwd IAT Min
# 31 Bwd IAT Tot 32 Bwd IAT Mean 33 Bwd IAT Std 34 Bwd IAT Max 35 Bwd IAT Min 36 Fwd PSH Flags 37 Bwd PSH Flags 38 Fwd URG Flags 39 Bwd URG Flags 40 Fwd Header Len
# 41 Bwd Header Len 42 Fwd Pkts/s 43 Bwd Pkts/s 44 Pkt Len Min 45 Pkt Len Max 46 Pkt Len Mean 47 Pkt Len Std 48 Pkt Len Var 49 FIN Flag Cnt 50 SYN Flag Cnt
# 51 RST Flag Cnt 52 PSH Flag Cnt 53 ACK Flag Cnt 54 URG Flag Cnt 55 CWE Flag Count 56 ECE Flag Cnt 57 Down/Up Ratio 58 Pkt Size Avg 59 Fwd Seg Size Avg 60 Bwd Seg Size Avg
# 61 Fwd Byts/b Avg 62 Fwd Pkts/b Avg 63 Fwd Blk Rate Avg 64 Bwd Byts/b Avg 65 Bwd Pkts/b Avg 66 Bwd Blk Rate Avg 67 Subflow Fwd Pkts 68 Subflow Fwd Byts 69 Subflow Bwd Pkts 70 Subflow Bwd Byts
# 71 Init Fwd Win Byts 72 Init Bwd Win Byts 73 Fwd Act Data Pkts 74 Fwd Seg Size Min 75 Active Mean 76 Active Std 77 Active Max 78 Active Min 79 Idle Mean 80 Idle Std
# 81 Idle Max 82 Idle Min 83 Label
#[1, 2, 3, 4, 6, 7, 12, 15, 16, 18, 24, 68, 71, 72]
def FranceDataset_data_Reader(inputFeatures):
    #print(inputFeatures)
    flow_dis = []
    features_name = []
    original_data = []
    file_index = []
    g = os.walk(r"./.gitignore/FranceDataset")
    for path, dir_list, file_list in g:
        print('Len of file_list ', len(file_list))
        for file_path in file_list:
            file_name = os.path.join(path, file_path)
            with open(file_name, 'r', encoding="utf-8") as csvfile:

                reader = csv.reader(csvfile)

                for index, row in enumerate(reader):

                    distance_list = []
                    if index == 0:
                        if len(original_data)==0:
                            original_data.append(row)
                        # for rowID, key in enumerate(row):
                        #     print(rowID, key,end =" ")
                        for rowID, key in enumerate(row):
                            
                            if rowID in inputFeatures and key not in features_name:
                                features_name.append(key)
                        continue
                    
                    for rowID, key in enumerate(row):
                        original_data.append(row)
                        if rowID not in inputFeatures:
                            continue
                        if rowID == 0:  ###192.168.20.14-18.184.187.186-30753-1883-6###
                            split_key(distance_list, key, '\-|\.')
                            #print(distance_list)
                        elif rowID == 1:  ###192.168.20.48###
                            IPProcessing(distance_list, key, '\.')
                            #print(distance_list)
                        elif rowID == 3:  ###52.94.241.146###
                            IPProcessing(distance_list, key, '\.')
                            #print(distance_list)
                        elif rowID == 6:  ### 6/8/2022  10:01:23 PM###
                            FranceDatasetTimeProcessing(distance_list, key, '\/|\  |\ |:|P|A|M')
                            #print(distance_list)
                        elif rowID == 83:
                            
                            continue
                        else:
                            distance_list.append(float(key))
                    
                    flow_dis.append(distance_list)
                #print("file len ", len(flow_dis))
                file_index.append(len(flow_dis))
                # if len(file_index) == 0:
                #     file_index.append(len(flow_dis))
                # else:
                #     file_index.append(len(flow_dis)-file_index[-1])
                #print(file_index)
                # print("flow list len ", len(flow_dis))
                # for key in flow_dis:
                #     print(key)
                #     print(len(key))
                # last_timestemp = datetime.strptime(row[6], '%m/%d/%Y %I:%M:%S %p')
    if 'Label' in features_name:
        features_name.remove('Label')
    return flow_dis, features_name, original_data, file_index


def FranceDataset_data_Reader_AM_PM(inputFeatures):
    # print(inputFeatures)
    flow_dis = []
    features_name = []
    original_data = []
    file_index = []
    g = os.walk(r"./.gitignore/FranceDataset")
    for path, dir_list, file_list in g:
        print('Len of file_list ', len(file_list))
        for index, file_path in enumerate(file_list):
            if index <= 80:
                continue
            file_name = os.path.join(path, file_path)
            with open(file_name, 'r', encoding="utf-8") as csvfile:

                reader = csv.reader(csvfile)

                for index, row in enumerate(reader):

                    distance_list = []
                    if index == 0:
                        if len(original_data) == 0:
                            original_data.append(row)
                        # for rowID, key in enumerate(row):
                        #     print(rowID, key,end =" ")
                        for rowID, key in enumerate(row):

                            if rowID in inputFeatures and key not in features_name:
                                features_name.append(key)
                        continue
                    date = ''
                    for rowID, key in enumerate(row):
                        original_data.append(row)
                        if rowID not in inputFeatures:
                            continue
                        if rowID == 0:  ###192.168.20.14-18.184.187.186-30753-1883-6###
                            split_key(distance_list, key, '\-|\.')
                            # print(distance_list)
                        elif rowID == 1:  ###192.168.20.48###
                            IPProcessing(distance_list, key, '\.')
                            # print(distance_list)
                        elif rowID == 3:  ###52.94.241.146###
                            IPProcessing(distance_list, key, '\.')
                            # print(distance_list)
                        elif rowID == 6:  ### 6/8/2022  10:01:23 PM###
                            date = key.split(" ")[0]
                            #print(key, date)
                            FranceDatasetTimeProcessing(distance_list, key, '\/|\  |\ |:|P|A|M')
                            # print(distance_list)
                        elif rowID == 83:

                            continue
                        else:
                            distance_list.append(float(key))
                    # print(row)
                    if 'AM' in row[6]:
                        distance_list.append(date+" AM")

                    elif'PM' in row[6]:
                        distance_list.append(date+' PM')

                    flow_dis.append(distance_list)
                # print("file len ", len(flow_dis))
                file_index.append(len(flow_dis))
                # if len(file_index) == 0:
                #     file_index.append(len(flow_dis))
                # else:
                #     file_index.append(len(flow_dis)-file_index[-1])
                # print(file_index)
                # print("flow list len ", len(flow_dis))
                # for key in flow_dis:
                #     print(key)
                #     print(len(key))
                # last_timestemp = datetime.strptime(row[6], '%m/%d/%Y %I:%M:%S %p')
    if 'Label' not in features_name:
         features_name.add('Label')
    return flow_dis, features_name, original_data, file_index
# UnicaucaDataset
# 0 Flow.ID 1 Source.IP 2 Source.Port 3 Destination.IP 4 Destination.Port 5 Protocol 6 Timestamp 7 Flow.Duration 8 Total.Fwd.Packets 9 Total.Backward.Packets 10 Total.Length.of.Fwd.Packets
# 11 Total.Length.of.Bwd.Packets 12 Fwd.Packet.Length.Max 13 Fwd.Packet.Length.Min 14 Fwd.Packet.Length.Mean 15 Fwd.Packet.Length.Std 16 Bwd.Packet.Length.Max 17 Bwd.Packet.Length.Min 18 Bwd.Packet.Length.Mean 19 Bwd.Packet.Length.Std 20 Flow.Bytes.s
# 21 Flow.Packets.s 22 Flow.IAT.Mean 23 Flow.IAT.Std 24 Flow.IAT.Max 25 Flow.IAT.Min 26 Fwd.IAT.Total 27 Fwd.IAT.Mean 28 Fwd.IAT.Std 29 Fwd.IAT.Max 30 Fwd.IAT.Min
# 31 Bwd.IAT.Total 32 Bwd.IAT.Mean 33 Bwd.IAT.Std 34 Bwd.IAT.Max 35 Bwd.IAT.Min 36 Fwd.PSH.Flags 37 Bwd.PSH.Flags 38 Fwd.URG.Flags 39 Bwd.URG.Flags 40 Fwd.Header.Length
# 41 Bwd.Header.Length 42 Fwd.Packets.s 43 Bwd.Packets.s 44 Min.Packet.Length 45 Max.Packet.Length 46 Packet.Length.Mean 47 Packet.Length.Std 48 Packet.Length.Variance 49 FIN.Flag.Count 50 SYN.Flag.Count
# 51 RST.Flag.Count 52 PSH.Flag.Count 53 ACK.Flag.Count 54 URG.Flag.Count 55 CWE.Flag.Count 56 ECE.Flag.Count 57 Down.Up.Ratio 58 Average.Packet.Size 59 Avg.Fwd.Segment.Size 60 Avg.Bwd.Segment.Size
# 61 Fwd.Header.Length.1 62 Fwd.Avg.Bytes.Bulk 63 Fwd.Avg.Packets.Bulk 64 Fwd.Avg.Bulk.Rate 65 Bwd.Avg.Bytes.Bulk 66 Bwd.Avg.Packets.Bulk 67 Bwd.Avg.Bulk.Rate 68 Subflow.Fwd.Packets 69 Subflow.Fwd.Bytes 70 Subflow.Bwd.Packets
# 71 Subflow.Bwd.Bytes 72 Init_Win_bytes_forward 73 Init_Win_bytes_backward 74 act_data_pkt_fwd 75 min_seg_size_forward 76 Active.Mean 77 Active.Std 78 Active.Max 79 Active.Min 80 Idle.Mean
# 81 Idle.Std 82 Idle.Max 83 Idle.Min 84 Label 85 L7Protocol 86 ProtocolName

# 24Flow.IAT.Max 3Destination.IP 2Source.Port 1Source.IP Fwd.IAT 73Init_Win_Bytes_Backward 4Destination.Port 12Fwd.Packet.length.Max 6Timestamp 69Subflow.Fwd.Bytes 72Init_Win_Bytes_Forward 
# 18Bwd.Packet.Length.Mean 16Bwd.Packet.Length.Max 15Fwd.Packet.Length.Std 7Flow.Duration
#[1, 2 ,3 , 4, 6, 7, 12, 15, 16, 18, 24, 69, 72, 73]
def UnicaucaDataset_data_Reader(inputFeatures):
    
    flow_dis = []
    original_data = []
    file_name = "./.gitignore/Dataset-Unicauca-Version2-87Atts.csv"
    features_name = []
    print("Inputfeatures ", len(inputFeatures))
    with open(file_name, 'r', encoding="utf-8") as csvfile:

        reader = csv.reader(csvfile)

        for index, row in enumerate(reader):
            original_data.append(row)
            distance_list = []
            if index == 0:
                # for rowID, key in enumerate(row):
                #     print(rowID, key,end =" ")
                for rowID, key in enumerate(row):
                    if rowID in inputFeatures and key not in features_name:
                        features_name.append(key)
                # print(len(features_name), features_name)
                continue
            date = ''
            month = ''
            for rowID, key in enumerate(row):
                if rowID not in inputFeatures:
                    continue
                if rowID == 0:  ###Flow ID 192.168.20.14-18.184.187.186-30753-1883-6### 
                    split_key(distance_list, key, '\-|\.')
                    # print("row0", distance_list)
                elif rowID == 1:  ### Source IP 192.168.20.48###
                    IPProcessing(distance_list, key, '\.')
                    # print("row1", distance_list) 
                elif rowID == 3:  ###Destination IP 52.94.241.146###
                    IPProcessing(distance_list, key, '\.')
                    # print("row3", distance_list) 
                elif rowID == 6:  ###Time stamp 6/8/2022  10:01:23 PM###

                    date = key.split("/")[0]
                    month = key.split("/")[1]
                    tlabel = UnicaucaDatasetTimeProcessing(distance_list, key, '\/|\  |\ |:|P|A|M')
                elif rowID == 84:
                    distance_list.append(key)
                elif rowID == 86:
                    distance_list.append(key)
                #     continue
                else:
                    # print(features_name[rowID])
                    # print("rowID", rowID, "key", key)
                    distance_list.append(float(key))

            distance_list.append(month+"/"+date)
            distance_list.append(tlabel)
            flow_dis.append(distance_list)
        # for key in flow_dis:
        #     print(key)
        #     print(len(key))
        # last_timestemp = datetime.strptime(row[6], '%m/%d/%Y %I:%M:%S %p')
        # if 'Label' in features_name:
        #     features_name.remove('Label')
        # if 'L7Protocol' in features_name:
        #     features_name.remove('L7Protocol')
        # if 'ProtocolName' in features_name:
        #     features_name.remove('ProtocolName')

        # print("flow_dis")
        # print(flow_dis)

    return flow_dis, features_name, original_data