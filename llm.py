import torch
import re
import itertools
import json
import os
from collections import Counter
from pprint import pprint
import yaml
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import time
import h5py
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
from itertools import takewhile
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#openai.api_key='sk-proj-7Z81mV19yyGYftz6OkU8T3BlbkFJZUUKdJA3WDkh2CaNhpZX'

# def prepare_questions(annotations):
#     """ Filter, Normalize and Tokenize question using LLM. """
#     prepared = []
#     questions = [q['question'] for q in annotations]

#     for question in questions:
#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo",
#             prompt=f"Clean and tokenize the following question: {question}",
#             max_tokens=50
#         )
#         clean_question = response.choices[0].text.strip()
#         question_list = clean_question.split(' ')
#         question_list = list(filter(None, question_list))
#         prepared.append(question_list)

#     return prepared


# def prepare_answers(annotations):
#     answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
#     prepared = []

#     for sample_answers in answers:
#         prepared_sample_answers = []
#         for answer in sample_answers:
#             response = openai.Completion.create(
#                 engine="gpt-3.5-turbo",
#                 prompt=f"Clean and tokenize the following answer: {answer}",
#                 max_tokens=50
#             )
#             clean_answer = response.choices[0].text.strip()
#             prepared_sample_answers.append(clean_answer)

#         prepared.append(prepared_sample_answers)
#     return prepared

# def generate_question_from_context(context):
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo",
#         prompt=f"Generate a question based on the following context: {context}",
#         max_tokens=50
#     )
#     question = response.choices[0].text.strip()
#     return question

# def generate_answer_from_context_and_question(context, question):
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo",
#         prompt=f"Generate an answer based on the following context: {context} and question: {question}",
#         max_tokens=50
#     )
#     answer = response.choices[0].text.strip()
#     return answer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def prepare_questions(annotations):
    """ Filter, Normalize, and Tokenize questions using GPT-2. """
    prepared = []
    questions = [q['question'] for q in annotations]

    for question in questions:
        # Tokenize and clean the question
        tokens = tokenizer.encode(question, return_tensors="pt")
        clean_question = tokenizer.decode(tokens[0], skip_special_tokens=True)
        prepared.append(clean_question.split())

    return prepared

def prepare_answers(annotations):
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
    prepared = []

    for sample_answers in answers:
        prepared_sample_answers = []
        for answer in sample_answers:
            # Tokenize and clean the answer
            tokens = tokenizer.encode(answer, return_tensors="pt")
            clean_answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
            prepared_sample_answers.append(clean_answer.split())

        prepared.append(prepared_sample_answers)
    return prepared

def generate_question_from_context(context):
    # Generate a question based on the context
    print(context,type(context))
    context=" ".join(context)
    input_ids = tokenizer.encode("Generate a question based on the following context: " + context, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    question = tokenizer.decode(output[0], skip_special_tokens=True)
    return question.strip()

def generate_answer_from_context_and_question(context, question):
    # Generate an answer based on the context and question
    input_ids = tokenizer.encode("Generate an answer based on the following context: " + context + " and question: " + question, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.strip()

def create_question_vocab(questions, min_count=0):
    """
    Extract vocabulary used to tokenize and encode questions.
    """
    words = itertools.chain.from_iterable([q for q in questions])  # chain('ABC', 'DEF') --> A B C D E F
    counter = Counter(words)

    counted_words = counter.most_common()
    # select only the words appearing at least min_count
    selected_words = list(takewhile(lambda x: x[1] >= min_count, counted_words))

    vocab = {t[0]: i for i, t in enumerate(selected_words, start=1)}

    return vocab


def create_answer_vocab(annotations, top_k):
    answers = itertools.chain.from_iterable(prepare_answers(annotations))

    counter = Counter(answers)
    counted_ans = counter.most_common(top_k)
    # start from labels from 0
    vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}

    return vocab

def create_vocab():
    config_path = 'config/default.yaml'
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    pprint(config)

    dir_path = config['annotations']['dir']
    output_dir = './prepro_data/'
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(dir_path, config['training']['train_split'] + '.json')
    with open(train_path, 'r', encoding='utf-8') as fd:
        train_ann = json.load(fd)

    questions = prepare_questions(train_ann)
    question_vocab = create_question_vocab(questions, config['annotations']['min_count_word'])
    answer_vocab = create_answer_vocab(train_ann, config['annotations']['top_ans'])

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }

    with open(config['annotations']['path_vocabs'], 'w') as fd:
        json.dump(vocabs, fd)

    print("vocabs saved in {}".format(config['annotations']['path_vocabs']))
    
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class ImageDataset(data.Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        # Load the paths to the images available in the folder
        self.image_names = self._load_img_paths()

        if len(self.image_names) == 0:
            raise (RuntimeError("Found 0 images in " + path + "\n"
                                                              "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        else:
            print('Found {} images in {}'.format(len(self), self.path))

    def __getitem__(self, index):
        item = {}
        item['name'] = self.image_names[index]
        item['path'] = os.path.join(self.path, item['name'])

        # Use PIL to load the image
        item['visual'] = Image.open(item['path']).convert('RGB')
        if self.transform is not None:
            item['visual'] = self.transform(item['visual'])

        return item

    def __len__(self):
        return len(self.image_names)

    def _load_img_paths(self):
        images = []
        for name in os.listdir(self.path):
            if is_image_file(name):
                images.append(name)
        return images


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        # TODO : Compute mean and std of VizWiz
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])





class NetFeatureExtractor(nn.Module):

    def __init__(self):
        super(NetFeatureExtractor, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        def save_att_features(module, input, output):
            self.att_feat = output

        def save_noatt_features(module, input, output):
            self.no_att_feat = output

        self.model.layer4.register_forward_hook(save_att_features)
        self.model.avgpool.register_forward_hook(save_noatt_features)

    def forward(self, x):
        self.model(x)
        return self.no_att_feat, self.att_feat


def feature_ext():
    # Load config yaml file
    config_path = 'config/default.yaml'

    if config_path is not None:
        with open(config_path, 'r') as handle:
            config = yaml.safe_load(handle)
            config = config['images']

    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = NetFeatureExtractor().to(device)
    net.eval()

    transform = get_transform(config['img_size'])
    dataset = ImageDataset(config['dir_demo'], transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['preprocess_batch_size'],
        num_workers=config['preprocess_data_workers'],
        shuffle=False,
        pin_memory=True,
    )

    h5_file = h5py.File(config['path_demo'], 'w')

    dummy_input = Variable(torch.ones(1, 3, config['img_size'], config['img_size']), volatile=True).to(device)
    _, dummy_output = net(dummy_input)

    att_features_shape = (
        len(data_loader.dataset),
        dummy_output.size(1),
        dummy_output.size(2),
        dummy_output.size(3)
    )

    noatt_features_shape = (
        len(data_loader.dataset),
        dummy_output.size(1)
    )

    h5_att = h5_file.create_dataset('att', shape=att_features_shape, dtype='float16')
    h5_noatt = h5_file.create_dataset('noatt', shape=noatt_features_shape, dtype='float16')

    dt = h5py.special_dtype(vlen=str)
    img_names = h5_file.create_dataset('img_name', shape=(len(data_loader.dataset),), dtype=dt)

    begin = time.time()

    print('Extracting features ...')
    idx = 0
    delta = config['preprocess_batch_size']

    for i, inputs in enumerate(tqdm(data_loader)):
        inputs_img = Variable(inputs['visual'].to(device, non_blocking=True), volatile=True)
        no_att_feat, att_feat = net(inputs_img)

        no_att_feat = no_att_feat.view(-1, 2048)

        h5_noatt[idx:idx + delta] = no_att_feat.data.cpu().numpy().astype('float16')
        h5_att[idx:idx + delta, :, :] = att_feat.data.cpu().numpy().astype('float16')
        img_names[idx:idx + delta] = inputs['name']

        idx += delta

    h5_file.close()

    end = time.time() - begin

    print('Finished in {}m and {}s'.format(int(end / 60), int(end % 60)))
    print('Created file : ' + config['path_demo'])





class Model(nn.Module):

    def __init__(self, config, num_tokens):
        super(Model, self).__init__()

        dim_v = config['model']['pooling']['dim_v']
        dim_q = config['model']['pooling']['dim_q']
        dim_h = config['model']['pooling']['dim_h']

        n_glimpses = config['model']['attention']['glimpses']

        self.text = TextEncoder(
            num_tokens=num_tokens,
            emb_size=config['model']['seq2vec']['emb_size'],
            dim_q=dim_q,
            drop=config['model']['seq2vec']['dropout'],
        )
        self.attention = Attention(
            dim_v=dim_v,
            dim_q=dim_q,
            dim_h=config['model']['attention']['mid_features'],
            n_glimpses=n_glimpses,
            drop=config['model']['attention']['dropout'],
        )
        self.classifier = Classifier(
            dim_input=n_glimpses * dim_v + dim_q,
            dim_h=dim_h,
            top_ans=config['annotations']['top_ans'],
            drop=config['model']['classifier']['dropout'],
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):

        q = self.text(q, list(q_len.data))
        # L2 normalization on the depth dimension
        v = F.normalize(v, p=2, dim=1)
        attention_maps = self.attention(v, q)
        v = apply_attention(v, attention_maps)
        # concatenate attended features and encoded question
        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, dim_input, dim_h, top_ans, drop=0.0, num_layers=3):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(dim_input, dim_h))
        self.add_module('relu1', nn.ReLU())
        
        for i in range(num_layers - 1):
            self.add_module(f'drop{i+2}', nn.Dropout(drop))
            self.add_module(f'lin{i+2}', nn.Linear(dim_h, dim_h))
            self.add_module(f'relu{i+2}', nn.ReLU())
        
        self.add_module('drop_last', nn.Dropout(drop))
        self.add_module('lin_last', nn.Linear(dim_h, top_ans))
        
class TextEncoder(nn.Module):
    def __init__(self, num_tokens, emb_size, dim_q, drop=0.0, num_layers=3):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size, padding_idx=0)
        self.dropout = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=dim_q,
                            num_layers=num_layers,  # Increased the number of LSTM layers
                            dropout=drop,  # Apply dropout between LSTM layers
                            batch_first=True)
        self.dim_q = dim_q

        # Initialize parameters
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.dropout(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (c, _) = self.lstm(packed)
        return c[-1]  


class Attention(nn.Module):
    def __init__(self, dim_v, dim_q, dim_h, n_glimpses, drop=0.0, num_layers=3):
        super(Attention, self).__init__()
        self.conv_v = nn.Conv2d(dim_v, dim_h, 1, bias=False)
        self.fc_q = nn.Linear(dim_q, dim_h)
        
        # Adding multiple linear layers with dropout
        self.fc_layers = nn.Sequential()
        for i in range(num_layers - 1):
            self.fc_layers.add_module(f'fc_{i}', nn.Linear(dim_h, dim_h))
            self.fc_layers.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            self.fc_layers.add_module(f'dropout_{i}', nn.Dropout(drop))

        self.conv_x = nn.Conv2d(dim_h, n_glimpses, 1)
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.conv_v(self.dropout(v))
        q = self.fc_q(self.dropout(q))
        q = repeat_encoded_question(q, v)
        x = self.relu(v + q)
        
        for layer in self.fc_layers:
            x = layer(x)

        x = self.conv_x(self.dropout(x))
        return x


def repeat_encoded_question(q, v):
    batch_size, h = q.size()
    q_tensor = q.view(batch_size, h, *([1, 1])).expand_as(v)
    return q_tensor


def apply_attention(v, attention):
    batch_size, spatial_vec_size = v.size()[:2]
    glimpses = attention.size(1)
    v = v.view(batch_size, spatial_vec_size, -1)
    attention = attention.view(batch_size, glimpses, -1)
    n_image_regions = v.size(2)

    attention = attention.view(batch_size * glimpses, -1) 
    attention = F.softmax(attention, dim=1)
    target_size = [batch_size, glimpses, spatial_vec_size, n_image_regions]
    v = v.view(batch_size, 1, spatial_vec_size, n_image_regions).expand(
        *target_size)  
    attention = attention.view(batch_size, glimpses, 1, n_image_regions).expand(
        *target_size) 
    weighted = v * attention
    weighted_mean = weighted.sum(dim=3)  
    return weighted_mean.view(batch_size, -1) 

def vqa_accuracy(predicted, true):
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    return (agreeing * 0.33333).clamp(max=1) 


class Tracker:

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value


def get_id_from_name(name):
    import re

    n = re.search('VizWiz_(.+?)_', name)
    if n:
        split = n.group(1)

    m = re.search(('VizWiz_%s_(.+?).jpg' % split), name)
    if m:
        found = m.group(1)

    return int(found)




class FeaturesDataset(data.Dataset):

    def __init__(self, features_path, mode):
        self.path_hdf5 = features_path

        assert os.path.isfile(self.path_hdf5), \
            'File not found in {}, you must extract the features first with images_preprocessing.py'.format(
                self.path_hdf5)

        self.hdf5_file = h5py.File(self.path_hdf5, 'r')
        self.dataset_features = self.hdf5_file[mode]  # noatt or att (attention)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset_features[index].astype('float32'))

    def __len__(self):
        return self.dataset_features.shape[0]


def get_loader(config, split):
    split = VQADataset(
        config,
        split
    )

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config['training']['batch_size'],
        shuffle=True if split == 'train' or split == 'trainval' else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config['training']['data_workers'],
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    batch.sort(key=lambda x: x['q_length'], reverse=True)
    return data.dataloader.default_collate(batch)


class VQADataset(data.Dataset):

    def __init__(self, config, split):
        super(VQADataset, self).__init__()

        with open(config['annotations']['path_vocabs'], 'r', encoding='utf-8') as fd:
            vocabs = json.load(fd)

        annotations_dir = config['annotations']['dir']

        path_ann = os.path.join(annotations_dir, split + ".json")
        with open(path_ann, 'r', encoding='utf-8') as fd:
            self.annotations = json.load(fd)

        self.max_question_length = config['annotations']['max_length']
        self.split = split
        self.vocabs = vocabs
        self.token_to_index = self.vocabs['question']
        self.answer_to_index = self.vocabs['answer']

        self.questions = prepare_questions(self.annotations)
        self.questions = [generate_question_from_context(q) for q in
                          self.questions]  # encode questions and return question and question lenght

        if self.split != 'test' and self.split != 'demo':
            self.answers = prepare_answers(self.annotations)
            self.answers = [generate_answer_from_context_and_question(a) for a in
                            self.answers]  # create a sparse vector of len(self.answer_to_index) for each question containing the occurances of each answer

        if self.split == "train" or self.split == "trainval":
            self._filter_unanswerable_samples()

        with h5py.File(config['images']['path_demo'], 'r') as f:
            img_names = f['img_name'][()]
        self.name_to_id = {name: i for i, name in enumerate(img_names)}

        self.img_names = [s['image'] for s in self.annotations]
        
        self.features = FeaturesDataset(config['images']['path_demo'], config['images']['mode'])

    def _filter_unanswerable_samples(self):
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])

                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1 

    def __getitem__(self, i):

        item = {}
        item['question'], item['q_length'] = self.questions[i]
        if (self.split != 'test' and self.split != 'demo'):
            item['answer'] = self.answers[i]
        img_name = self.img_names[i]
        img_name = img_name.encode('utf-8') 
        print(self.name_to_id)
        feature_id = self.name_to_id[img_name]
        print("done")
        item['img_name'] = self.img_names[i]
        item['visual'] = self.features[feature_id]
        item['sample_id'] = i

        return item

    def __len__(self):
        return len(self.questions)

def train(model, loader, optimizer, tracker, epoch, split):
    model.train()

    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for item in tq:
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v)
        q = Variable(q)
        a = Variable(a)
        q_length = Variable(q_length)

        out = model(v, q, q_length)

        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = vqa_accuracy(out.data, a.data).cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    # Print average accuracy for the epoch
    print(f'Epoch {epoch} - {split} Accuracy: {acc_tracker.mean.value:.4f}')

def evaluate(model, loader, tracker, epoch, split):
    model.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}

    predictions = []
    samples_ids = []
    accuracies = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    with torch.no_grad():
        for item in tq:
            v = item['visual']
            q = item['question']
            a = item['answer']
            sample_id = item['sample_id']
            q_length = item['q_length']

            v = Variable(v)
            q = Variable(q)
            a = Variable(a)
            q_length = Variable(q_length)

            out = model(v, q, q_length)

            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = vqa_accuracy(out.data, a.data).cpu()

            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        predictions.append(out.data.cpu().numpy())
        samples_ids.append(sample_id.cpu().numpy())
        accuracies.append(acc.mean().item())
    eval_results = {
        'answers': predictions,
        'accuracies': accuracies,
        'samples_ids': samples_ids,
        'avg_accuracy': acc_tracker.mean.value,
        'avg_loss': loss_tracker.mean.value
    }
    

    # Print average accuracy for the epoch
    print(f'Epoch {epoch} - {split} Accuracy: {acc_tracker.mean.value:.4f}')
    return eval_results






def predict_answers(model, loader, split):
    model.eval()
    predicted = []
    samples_ids = []

    tq = tqdm(loader)

    print("Evaluating...\n")

    for item in tq:
        v = item['visual']
        q = item['question']
        sample_id = item['sample_id']
        q_length = item['q_length']

        v = Variable(v)
        q = Variable(q)
        q_length = Variable(q_length)

        out = model(v, q, q_length)

        _, answer = out.data.cpu().max(dim=1)

        predicted.append(answer.view(-1))
        samples_ids.append(sample_id.view(-1).clone())

    predicted = list(torch.cat(predicted, dim=0))
    samples_ids = list(torch.cat(samples_ids, dim=0))

    print("Evaluation completed")

    return predicted, samples_ids

def create_submission(input_annotations, predicted, samples_ids, vocabs):
    answers = torch.FloatTensor(predicted)
    indexes = torch.IntTensor(samples_ids)
    ans_to_id = vocabs['answer']
    # need to translate answers ids into answers
    id_to_ans = {idx: ans for ans, idx in ans_to_id.items()}
    # sort based on index the predictions
    '''sort_index = np.argsort(indexes)
    sorted_answers = np.array(answers, dtype='int_')[sort_index]'''
     # Sort predictions based on index
    sort_index = np.argsort(indexes.numpy())
    sorted_answers = answers[sort_index].numpy().astype(int)


    real_answers = []
    for ans_id in sorted_answers:
        ans = id_to_ans[ans_id]
        real_answers.append(ans)

    # Integrity check
    assert len(input_annotations) == len(real_answers)

    submission = []
    for i in range(len(input_annotations)):
        pred = {}
        pred['image'] = input_annotations[i]['image']
        pred['answer'] = real_answers[i]
        submission.append(pred)
        if (real_answers[i]!='unanswerable'):

        # Print image name, answer, and display image
            print("Image:", input_annotations[i]['image'])
            print("Question:", input_annotations[i]['question'])
            print("Answer:", real_answers[i])
        
        # Display the image
            # image_data = load_image(input_annotations[i]['image'])  # Load image data
            # Display the image if it was successfully loaded
            # if image_data is not None:
            #     plt.imshow(image_data)
            #     plt.axis('off')
            #     plt.show()
    
        
    return submission

def load_image(image_path):
    # Load image data based on the image_path
    image_path = './data/images/val/' + image_path

    try:
        with Image.open(image_path) as img:
            image_data = img.convert('RGB')  # Convert to RGB format if necessary
    except FileNotFoundError:
        print("Error: Image file not found at path:", image_path)
        return None
    except Exception as e:
        print("Error loading image:", e)
        return None
    return image_data

def test_pro():
    # Load config yaml file
    path_config = 'config/default.yaml'  # Path to the YAML config file

    with open(path_config, 'r') as handle:
        config = yaml.safe_load(handle)

    cudnn.benchmark = True

    # Generate dataset and loader
    print("Loading samples to predict from %s" % os.path.join(config['annotations']['dir'],
                                                              config['demo']['split'] + '.json'))

    # Load annotations
    path_annotations = os.path.join(config['annotations']['dir'], config['demo']['split'] + '.json')
    path_annotations = config['annotations']['dir']+'/'+config['demo']['split']+'.json'
    input_annotations = json.load(open(path_annotations, 'r'))

    # Data loader and dataset
    input_loader = get_loader(config, split=config['demo']['split'])

    config['demo']['model_path'] = './logs/vizwiz/pray/final_log.pth'
    # Load model weights
    print("Loading Model from %s" % config['demo']['model_path'])
    log = torch.load(config['demo']['model_path'], map_location=torch.device('cpu'))

    # Num tokens seen during training
    num_tokens = len(log['vocabs']['question']) + 1
    # Use the same configuration used during training
    train_config = log['config']

    model = nn.DataParallel(Model(train_config, num_tokens))

    dict_weights = log['weights']
    model.load_state_dict(dict_weights)

    predicted, samples_ids = predict_answers(model, input_loader, split=config['demo']['split'])

    submission = create_submission(input_annotations, predicted, samples_ids, input_loader.dataset.vocabs)

    with open(config['demo']['submission_file'], 'w') as fd:
        json.dump(submission, fd)

    print("Submission file saved in %s" % config['demo']['submission_file'])
def optimize():
    feature_ext()
    test_pro()

