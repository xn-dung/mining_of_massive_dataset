import numpy as np
import torch
import json


class file_loader:
    """Data loader for STDN model - PyTorch compatible"""
    
    def __init__(self, config_path="data.json"):
        self.config = json.load(open(config_path, "r"))
        # how many timeslots per day (48 here)
        self.timeslot_daynum = int(86400 / self.config["timeslot_sec"])
        self.threshold = int(self.config["threshold"])
        self.isVolumeLoaded = False
        self.isFlowLoaded = False

    def load_flow(self):
        """Load flow data"""
        self.flow_train = np.load(open(self.config["flow_train"], "rb"))["flow"] / self.config["flow_train_max"]
        self.flow_test = np.load(open(self.config["flow_test"], "rb"))["flow"] / self.config["flow_train_max"]
        self.isFlowLoaded = True

    def load_volume(self):
        """Load volume data"""
        # shape (timeslot_num, x_num, y_num, type=2)
        self.volume_train = np.load(open(self.config["volume_train"], "rb"))["volume"] / self.config["volume_train_max"]
        self.volume_test = np.load(open(self.config["volume_test"], "rb"))["volume"] / self.config["volume_train_max"]
        self.isVolumeLoaded = True

    def sample_stdn(self, datatype, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7,
                    hist_feature_daynum=7, last_feature_num=48, nbhd_size=1, cnn_nbhd_size=3, 
                    return_torch=True):
        """
        Sample data for STDN model
        
        :param datatype: 'train' or 'test'
        :param return_torch: if True, return PyTorch tensors instead of numpy arrays
        :return: (att_cnn_features, att_flow_features, att_lstm_features, 
                  cnn_features, flow_features, lstm_features, labels)
        """
        if not self.isVolumeLoaded:
            self.load_volume()

        if not self.isFlowLoaded:
            self.load_flow()

        if long_term_lstm_seq_len % 2 != 1:
            print("Att-lstm seq_len must be odd!")
            raise ValueError("Attention LSTM sequence length must be odd")

        if datatype == "train":
            data = self.volume_train
            flow_data = self.flow_train
        elif datatype == "test":
            data = self.volume_test
            flow_data = self.flow_test
        else:
            print("Please select **train** or **test**")
            raise ValueError("datatype must be 'train' or 'test'")

        cnn_att_features = []
        lstm_att_features = []
        flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features.append([])
            cnn_att_features.append([])
            flow_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])

        cnn_features = []
        flow_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
            flow_features.append([])
        short_term_lstm_features = []
        labels = []

        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = data.shape[0]
        volume_type = data.shape[-1]

        for t in range(time_start, time_end):
            if t % 100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    # sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        # cnn features, zero_padding
                        cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                        # actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                # boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                # get features
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = \
                                    data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

                        # flow features, 4 type
                        flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                        flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                        flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                        flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                        flow_feature = np.zeros(flow_feature_curr_in.shape + (4,))
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        
                        # calculate local flow, same shape cnn
                        local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                # boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = \
                                    flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                        flow_features[seqn].append(local_flow_feature)

                        # lstm features - neighborhood feature
                        nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                        for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                            for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                # boundary check
                                if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                    continue
                                nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = \
                                    data[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()

                        # last feature
                        last_feature = data[real_t - last_feature_num:real_t, x, y, :].flatten()

                        # hist feature
                        hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum:real_t:self.timeslot_daynum, x, y, :].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    # sample att-lstms
                    for att_lstm_cnt in range(att_lstm_num):
                        long_term_lstm_samples = []
                        att_t = t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + \
                                (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        
                        # att-lstm seq len
                        for seqn in range(long_term_lstm_seq_len):
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            # cnn features
                            cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = \
                                        data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            # flow features
                            flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                            flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                            flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                            flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                            flow_feature = np.zeros(flow_feature_curr_in.shape + (4,))
                            flow_feature[:, :, 0] = flow_feature_curr_out
                            flow_feature[:, :, 1] = flow_feature_curr_in
                            flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                            flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                            
                            local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = \
                                        flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            # att-lstm features - neighborhood feature
                            nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                            for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                                for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                    if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                        continue
                                    nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = \
                                        data[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()

                            # last feature
                            last_feature = data[real_t - last_feature_num:real_t, x, y, :].flatten()

                            # hist feature
                            hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum:real_t:self.timeslot_daynum, x, y, :].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    # label
                    labels.append(data[t, x, y, :].flatten())

        # Convert to arrays
        output_cnn_att_features = []
        output_flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                output_cnn_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])
        
        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        short_term_lstm_features = np.array(short_term_lstm_features)
        labels = np.array(labels)

        # Convert to PyTorch tensors if requested
        if return_torch:
            output_cnn_att_features = [torch.from_numpy(f).float() for f in output_cnn_att_features]
            output_flow_att_features = [torch.from_numpy(f).float() for f in output_flow_att_features]
            lstm_att_features = [torch.from_numpy(f).float() for f in lstm_att_features]
            cnn_features = [torch.from_numpy(f).float() for f in cnn_features]
            flow_features = [torch.from_numpy(f).float() for f in flow_features]
            short_term_lstm_features = torch.from_numpy(short_term_lstm_features).float()
            labels = torch.from_numpy(labels).float()

        return output_cnn_att_features, output_flow_att_features, lstm_att_features, \
               cnn_features, flow_features, short_term_lstm_features, labels
