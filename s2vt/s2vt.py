# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
#from model_RGB import get_video_train_data,get_video_test_data,preProBuildWordVocab
import pandas as pd
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import os
import re
    
class DataSet_MSVD():
    def __init__(self,csv_path,video_step,caption_step,image_dim,batch_size=50,video_train_data_path="./data/video_corpus.csv",\
                 video_train_feat_path="./rgb_train_features",video_test_data_path="./data/video_corpus.csv",\
                 video_test_feat_path="./rgb_test_features"):
        self.csv_path=csv_path;
        self.batch_size=batch_size;
        self.video_train_data_path = video_train_data_path;
        self.video_train_feat_path = video_train_feat_path;
        self.video_test_data_path = video_test_data_path;
        self.video_test_feat_path = video_test_feat_path;
        self.video_step = video_step;
        self.caption_step = caption_step;
        self.image_dim = image_dim;
        
        
        train_data = self.get_video_train_data(video_train_data_path, video_train_feat_path);
        train_captions = train_data['Description'].values;
        test_data = self.get_video_test_data(video_test_data_path, video_test_feat_path);
        test_captions = test_data['Description'].values;
        if not os.path.exists("./data/wordtoix.npy") or not os.path.exists('./data/ixtoword.npy') \
                                        or not os.path.exists("./data/bias_init_vector.npy"):
            captions_list = list(train_captions) + list(test_captions);
            captions = np.asarray(captions_list, dtype=np.object);
            
            #replace some characters to space, this means delete them
            captions = map(lambda x: x.replace('.', ''), captions);
            captions = map(lambda x: x.replace(',', ''), captions)
            captions = map(lambda x: x.replace('"', ''), captions)
            captions = map(lambda x: x.replace('\n', ''), captions)
            captions = map(lambda x: x.replace('?', ''), captions)
            captions = map(lambda x: x.replace('!', ''), captions)
            captions = map(lambda x: x.replace('\\', ''), captions)
            captions = map(lambda x: x.replace('/', ''), captions)
        
            #word to index, index to word,
            wordtoix, ixtoword, bias_init_vector = self.preProBuildWordVocab(captions, word_count_threshold=0)
            
            np.save("./data/wordtoix", wordtoix)
            np.save('./data/ixtoword', ixtoword)
            np.save("./data/bias_init_vector", bias_init_vector)
        else:
            wordtoix, ixtoword, bias_init_vector = np.load("./data/wordtoix.npy").tolist(),\
                                                   np.load('./data/ixtoword.npy').tolist(),\
                                                   np.load("./data/bias_init_vector.npy").tolist();
        #reset dataset
        current_train_data = self.reset_train_data(train_data);
        
        self.train_data = train_data;
        self.current_train_data = current_train_data;
        self.test_data = test_data;
        self.word2idx = wordtoix;
        self.idx2word = ixtoword;
        self.bias_init_vector = bias_init_vector;
        
        self.minibatch_start=0;
        self.train_data_size = len(current_train_data);
        self.word_num = len(self.word2idx);
        
    def next_batch(self):
        if self.minibatch_start+self.batch_size<self.train_data_size:
            minibatch_data = self.current_train_data[self.minibatch_start:(self.minibatch_start+self.batch_size)];
            #read data
            video_data_path = minibatch_data["video_path"].values;
            video_features = map(lambda video_path: np.load(video_path),video_data_path);
            video_data = np.zeros([self.batch_size,self.video_step,self.image_dim]);
            for idx,feature in enumerate(video_features):
                video_data[idx,:len(feature)]=feature;
            
            caption_data = minibatch_data["Description"].values;
            caption_data = self.preprocess_caption(caption_data);
            
            self.minibatch_start += self.batch_size;
        else:
# =============================================================================
#             minibatch_data = self.current_train_data[self.minibatch_start:self.train_data_size];
#             minibatch_size1 = self.batch_size+self.minibatch_start-self.train_data_size;
#             #read data
#             video_data_path = minibatch_data["video_path"].values;
#             video_features = map(lambda video_path: np.load(video_path),video_data_path);
#             video_data = np.zeros([self.batch_size,self.video_step,self.image_dim]);
#             for idx,feature in enumerate(video_features):
#                 video_data[idx,:len(feature)]=feature;
#             
#             caption_data = minibatch_data["Description"].values;
#             caption_data = self.preprocess_caption(caption_data);
# =============================================================================
            
            self.current_train_data = self.reset_train_data(self.train_data);
            self.minibatch_start = 0;
            
            video_data = [];
            caption_data = [];
        
        return video_data,caption_data;
    
    
    def preprocess_caption(self,current_captions):
        current_captions = map(lambda x: '<bos> ' + x, current_captions)
        current_captions = map(lambda x: x.replace('.', ''), current_captions)
        current_captions = map(lambda x: x.replace(',', ''), current_captions)
        current_captions = map(lambda x: x.replace('"', ''), current_captions)
        current_captions = map(lambda x: x.replace('\n', ''), current_captions)
        current_captions = map(lambda x: x.replace('?', ''), current_captions)
        current_captions = map(lambda x: x.replace('!', ''), current_captions)
        current_captions = map(lambda x: x.replace('\\', ''), current_captions)
        current_captions = map(lambda x: x.replace('/', ''), current_captions)

        for idx, each_cap in enumerate(current_captions):
            word = each_cap.lower().split(' ')
            if len(word) < self.caption_step:
                current_captions[idx] = current_captions[idx] + ' <eos>'
            else:
                new_word = ''
                for i in range(self.caption_step-1):
                    new_word = new_word + word[i] + ' '
                current_captions[idx] = new_word + '<eos>'

        current_caption_ind = []
        for cap in current_captions:
            current_word_ind = []
            for word in cap.lower().split(' '):
                if word in self.word2idx:
                    current_word_ind.append(self.word2idx[word])
                else:
                    current_word_ind.append(self.word2idx['<unk>']);
            current_caption_ind.append(current_word_ind)

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=self.caption_step)
        current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
    
        return current_caption_matrix;

    def reset_train_data(self,train_data):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.loc[index];
        
        #group_data = train_data.groupby('video_path');
        #print group_data;
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True);
        return current_train_data;

    def get_video_train_data(self,video_data_path, video_feat_path):
        #read csv file using panda
        video_data = pd.read_csv(video_data_path, sep=',')
        #filte Language==English
        video_data = video_data[video_data['Language'] == 'English']
        #get file name VideoID_Start_End.avi.npy
        video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
        #add video path
        video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
        #filte video path exist
        video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
        #filte Description is a string (not none or other)
        video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
        #filte video is unique
        unique_filenames = sorted(video_data['video_path'].unique())
        train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
        return train_data

    def get_video_test_data(self,video_data_path, video_feat_path):
        video_data = pd.read_csv(video_data_path, sep=',')
        video_data = video_data[video_data['Language'] == 'English']
        video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
        video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
        video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
        video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    
        unique_filenames = sorted(video_data['video_path'].unique())
        test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
        return test_data
    
    def preProBuildWordVocab(self,sentence_iterator, word_count_threshold=5):
        # borrowed this function from NeuralTalk
        print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)
        word_counts = {}
        nsents = 0
        #statistic the word
        for sent in sentence_iterator:
            nsents += 1
            for w in sent.lower().split(' '):
               word_counts[w] = word_counts.get(w, 0) + 1
        #filte the word whose number is lower than the threshold
        #vocab is a list containing all words which is more than threshold
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        print 'filtered words from %d to %d' % (len(word_counts), len(vocab))
    
        ixtoword = {}
        ixtoword[0] = '<pad>'
        ixtoword[1] = '<bos>'
        ixtoword[2] = '<eos>'
        ixtoword[3] = '<unk>'
    
        wordtoix = {}
        wordtoix['<pad>'] = 0
        wordtoix['<bos>'] = 1
        wordtoix['<eos>'] = 2
        wordtoix['<unk>'] = 3
    
        #get the index and content, that is (idx,word)
        for idx, w in enumerate(vocab):
            wordtoix[w] = idx+4
            ixtoword[idx+4] = w
    
        word_counts['<pad>'] = nsents
        word_counts['<bos>'] = nsents
        word_counts['<eos>'] = nsents
        word_counts['<unk>'] = nsents
    
        #how to use this variable? bias_init_vector?
        bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    
        return wordtoix, ixtoword, bias_init_vector

class VideoCaption(nn.Module):
    def __init__(self,batch_size,image_dim,word_num,img_embed_dim,word_embed_dim,hidden_dim,video_step,caption_step):
        super(VideoCaption,self).__init__();
        
        self.batch_size = batch_size;
        self.image_dim = image_dim;
        self.img_embed_dim = img_embed_dim;
        self.word_embed_dim = word_embed_dim;
        self.hidden_dim = hidden_dim;
        self.video_step = video_step;
        self.caption_step = caption_step;
        self.word_num = word_num;
        self.word2vec = nn.Embedding(word_num,word_embed_dim);
        
        nn.init.uniform(self.word2vec.weight,-0.1,0.1);
        self.vec2word = nn.Linear(hidden_dim,word_num);
        nn.init.uniform(self.vec2word.weight,-0.1,0.1);
        nn.init.constant(self.vec2word.bias,0);
        self.img_embed = nn.Linear(image_dim,img_embed_dim);
        nn.init.uniform(self.img_embed.weight,-0.1,0.1);
        nn.init.constant(self.img_embed.bias,0);
        self.lstm1 = nn.LSTMCell(input_size=img_embed_dim,hidden_size=hidden_dim);
        #nn.init.uniform(self.lstm1.weight_hh,-0.1,0.1);
        #nn.init.uniform(self.lstm1.weight_ih,-0.1,0.1);
        nn.init.orthogonal(self.lstm1.weight_hh);
        nn.init.orthogonal(self.lstm1.weight_ih);
        
        self.lstm2 = nn.LSTMCell(input_size=word_embed_dim+hidden_dim, hidden_size=hidden_dim);
        #nn.init.uniform(self.lstm2.weight_hh,-0.1,0.1);
        #nn.init.uniform(self.lstm2.weight_ih,-0.1,0.1);
        nn.init.orthogonal(self.lstm2.weight_hh);
        nn.init.orthogonal(self.lstm2.weight_ih);
        
    def forward(self, input_image, input_caption, caption_mask):
        '''
        input_image: int Variable, batch_size x video_step x image_dim
        input_caption: int Variable, batch_size x (1+caption_step) x 1 (word is idx, so the dim is 1)
        '''
        image_embeded_vector = self.img_embed(input_image);
        word_vec = self.word2vec(input_caption);
        
        #encoding
        state1 = Variable(torch.zeros(self.batch_size,self.lstm1.hidden_size)).cuda();
        state2 = Variable(torch.zeros(self.batch_size,self.lstm2.hidden_size)).cuda();
        output1 = Variable(torch.zeros(self.batch_size,self.lstm1.hidden_size)).cuda();
        output2 = Variable(torch.zeros(self.batch_size,self.lstm2.hidden_size)).cuda();
        padding_for_lstm1 = Variable(torch.zeros(self.batch_size,self.img_embed_dim)).cuda();
        padding_for_lstm2 = Variable(torch.zeros(self.batch_size,self.word_embed_dim)).cuda();
        
        for step in xrange(self.video_step):
            output1,state1 = self.lstm1(image_embeded_vector[:,step,:],(output1,state1));
            output2,state2 = self.lstm2(torch.cat((padding_for_lstm2,output1),1),(output2,state2));
            
        
        loss=Variable(torch.FloatTensor([0])).cuda();
        #decoding
        #one_hot_eye = np.eye(self.word_num).astype('int64');
        for step in xrange(self.caption_step):
            output1,state1 = self.lstm1(padding_for_lstm1,(output1, state1));
            output2,state2 = self.lstm2(torch.cat((word_vec[:,step,:],output1),1),(output2, state2));
            
            word_onehot = self.vec2word(output2);
            #word_onehot_softmax = nn.Softmax(dim=1)(word_onehot);
            labels = input_caption[:,step+1];
            
            #one_hot_labels = np.zeros((labels.data.shape[0],self.word_num),dtype='int64');
            #for idx,data in enumerate(labels.data):
             #   one_hot_labels[idx]=one_hot_eye[data];
            
            #labels_onehot = Variable(torch.FloatTensor(one_hot_labels)).cuda();
            #labels_onehot_list = labels_onehot.cpu().data.tolist()
            #print len(labels_onehot_list)
            #loss_func = nn.BCEWithLogitsLoss()*caption_mask[:,step];
            loss_func = nn.CrossEntropyLoss(reduce=False);
            
            loss_temp = loss_func(word_onehot,labels)*(caption_mask[:,step+1].float());
            loss += torch.sum(loss_temp)/self.batch_size;
        
        return loss;
    
    def generate_cpu(self, input_image):
        image_embeded_vector = self.img_embed(input_image);
        
        #encoding
        state1 = Variable(torch.zeros(1,self.lstm1.hidden_size));
        state2 = Variable(torch.zeros(1,self.lstm2.hidden_size));
        output1 = Variable(torch.zeros(1,self.lstm1.hidden_size));
        output2 = Variable(torch.zeros(1,self.lstm2.hidden_size));
        padding_for_lstm1 = Variable(torch.zeros(1,self.img_embed_dim));
        padding_for_lstm2 = Variable(torch.zeros(1,self.word_embed_dim));
        
        for step in xrange(self.video_step):
            output1,state1 = self.lstm1(image_embeded_vector[step,:],(output1,state1));
            output2,state2 = self.lstm2(torch.cat((padding_for_lstm2,output1),1),(output2,state2));
            
        
        words=[]
        #decoding
        #set '<bos>'
        previous_word = self.word2vec(Variable(torch.LongTensor([1])));
        for step in xrange(self.caption_step):
            output1,state1 = self.lstm1(padding_for_lstm1,(output1, state1));
            output2,state2 = self.lstm2(torch.cat((previous_word,output1),1),(output2, state2));
            #previous_word = output2;
            
            word_onehot = self.vec2word(output2);
            #print word_onehot.shape
            _,word_idx = torch.max(word_onehot,1);
            #print word_idx.data[0];
            
            words.append(word_idx.data[0]);
            
            previous_word = self.word2vec(word_idx);
        
        return words;

batch_size = 100;
image_dim = 4096;
img_embed_dim = 1000;
word_embed_dim = 1000;
hidden_dim = 1000;
video_step = 80;
caption_step = 20;
epoches=1001;
csv_path = "./data/video_corpus.csv"


def train(check_point=None):
    #parameters
    
    loss_log_file = "./loss.txt";
    data_set = DataSet_MSVD(csv_path=csv_path,video_step=video_step,caption_step=caption_step,\
                            image_dim=image_dim,batch_size=batch_size);
    word_num = data_set.word_num;
    video_caption_net = VideoCaption(batch_size=batch_size, image_dim=image_dim,\
                                     word_num = word_num, img_embed_dim=img_embed_dim,\
                                     word_embed_dim=word_embed_dim,\
                                     hidden_dim=hidden_dim,video_step=video_step,\
                                     caption_step=caption_step);
    if check_point != None:
        video_caption_net.load_state_dict(state_dict=torch.load(check_point));
        start_epoche = int(re.search(r"(\d*)\Z",check_point).groups()[-1]);
    else:
        start_epoche = int(0);
        
    video_caption_net.cuda();
    print video_caption_net;
    
    optimizer = torch.optim.Adam(video_caption_net.parameters(),lr=10e-4);
    
    loss_list=[];
    log_fp = open(loss_log_file,"w");
    #torch.backends.cudnn = False;

    for epoche in range(start_epoche,epoches):
        mini_batch_idx=0;
        while True:
            start_time = time.time();
            mini_batch_idx += 1;
            video_data,caption_data = data_set.next_batch();
            if len(video_data) == 0:
                break;
            
            caption_mask = np.zeros( (caption_data.shape[0], caption_data.shape[1]) )
            nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, caption_data ) )

            for ind, row in enumerate(caption_mask):
                row[:nonzeros[ind]] = 1
                
            video_data,caption_data,caption_mask = Variable(torch.FloatTensor(video_data)),Variable(torch.LongTensor(caption_data)),\
                                                        Variable(torch.LongTensor(caption_mask));
            video_data,caption_data,caption_mask = video_data.cuda(),caption_data.cuda(),caption_mask.cuda();
            loss = video_caption_net(video_data,caption_data,caption_mask);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            loss_list.append(loss.data[0]);
            
            print("epoche:{0},mini_batch:{1},loss:{2},Escape time:{3}".format(\
                    epoche,mini_batch_idx,loss.data[0],str(time.time()-start_time)));
            log_fp.write("epoche:{0},mini_batch:{1},loss:{2},Escape time:{3}\n".format(\
                    epoche,mini_batch_idx,loss.data[0],str(time.time()-start_time)));
        if epoche%10 == 0:
            torch.save(video_caption_net.state_dict(),"./model_temp/s2vt.pytorch.{0}".format(epoche));
            ax=plt.subplot(111)
            plt.plot(range(len(loss_list)),loss_list,color='black');
            #plt.plot(range(epoches),test_loss_list,color="red");
            plt.show();   
    
    #save model
    torch.save(video_caption_net.state_dict(),"./s2vt.pytorch");
             
    ax=plt.subplot(111)
    plt.plot(range(len(loss_list)),loss_list,color='black');
    #plt.plot(range(epoches),test_loss_list,color="red");
    plt.show();    
    log_fp.close();
    
    
def test(state_dict_path):
    #parameters

    data_set = DataSet_MSVD(csv_path=csv_path,video_step=video_step,caption_step=caption_step,\
                            image_dim=image_dim,batch_size=batch_size);
    #test 
    word_num = data_set.word_num;
    video_caption_net = VideoCaption(batch_size=batch_size, image_dim=image_dim,\
                                     word_num = word_num, img_embed_dim=img_embed_dim,\
                                     word_embed_dim=word_embed_dim,\
                                     hidden_dim=hidden_dim,video_step=video_step,\
                                     caption_step=caption_step);
    #video_caption_net.cuda();
    print video_caption_net;
    video_caption_net.load_state_dict(state_dict=torch.load(state_dict_path));
    
    #test data
    test_output_txt_fd = open("./test_result.txt","wb");
    test_data_path = data_set.test_data["video_path"].unique();
    for idx,data_path in enumerate(test_data_path):
        print("idx:{0},data_path:{1}".format(idx,data_path))
        video_feature = np.load(data_path);
        if video_feature.shape[0] != video_step:
            continue;
        video_feature = Variable(torch.FloatTensor(video_feature));
        #video_feature.cuda();
        words = video_caption_net.generate_cpu(video_feature);
        
        generated_words = [data_set.idx2word[word] for word in words];

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        print generated_sentence,'\n'
        test_output_txt_fd.write(data_path + '\n')
        test_output_txt_fd.write(generated_sentence + '\n\n')

if __name__=="__main__":
    #train(check_point=None);
    test("./model_temp/s2vt.pytorch.1000");

    
    

    