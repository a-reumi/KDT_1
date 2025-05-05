# %%
## 모듈 로딩
## - 데이터 분석 및 시각화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   # 진행상황 시각화 (프로그레스바 progressbar)

## 한국어_자연어_처리 형태소 분석기
from konlpy.tag import Okt         

## - Pytorch 관련
import torch

## - 데이터셋 : 학습용/검증용/테스트용 분리
from sklearn.model_selection import train_test_split

## - Vocab 생성 시 단어 빈도 처리 위한 python 기본 모듈
from collections import Counter

from nltk import FreqDist           ## 자연어 처리 

# %%
## 데이터 준비
DATA_FILE_TRAIN = '../data/open/train_data.csv'                                                                                                                                                                                                                                                                                                                                                                                                                                                              
DATA_FILE_TEST = '../data/open/test_data.csv'

# %%
train_df=pd.read_csv(DATA_FILE_TRAIN)
test_df=pd.read_csv(DATA_FILE_TEST)

# %%
train_df.info()

# %%
test_df.info()

# %%
print('결측값 여부 :',train_df.isnull().values.any())

# %%
print('레이블 개수')
print(train_df.groupby('title').size().reset_index(name='count')) 

# %%
X_data = train_df['title'].tolist()
y_data = train_df['topic_idx'].tolist()

# %% [markdown]
# ### 데이터셋 분리 : 학습용/검증용/테스트용

# %%
## 데이터셋 분리 : 학습용/검증용/테스트용
# 4. train / valid+test 먼저 나누기 (30%)

X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y_data)


# %%
# 5. valid / test 나누기 (15%씩)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                test_size=0.5, 
                                                random_state=42, 
                                                stratify=y_temp)

# %%
# 확인
print(f'학습 데이터 크기 : {len(X_train)}')
print(f'검증 데이터 크기 : {len(X_val)}')
print(f'테스트 데이터 크기 : {len(X_test)}')

# %% [markdown]
# ### 단어사전 생성 

# %%
from collections import Counter
from konlpy import *

okt      = Okt()
okt_morphs = [okt.morphs(sent) for sent in X_train]

# %%
# 토큰화 함수 정의
def tokenize(sentences):
    tokenized_sentences = []
    for sent in tqdm(sentences, desc="Tokenizing"):
        tokens = okt.morphs(sent)
        tokens = [word.lower() for word in tokens]
        # 불용어 제거, 원형복원, 구두점 추가해서 정리 
        tokenized_sentences.append(tokens)
    return tokenized_sentences


# %%
# 실제 토큰화 수행
tokenized_X_train = tokenize(X_train)
tokenized_X_val = tokenize(X_val)
tokenized_X_test = tokenize(X_test)

# %%
# 상위 샘플 2개 출력
for sent in tokenized_X_train[:2]:
  print(sent)

# %% [markdown]
# ### Vocab 생성 

# %%
## 단어 추출 후 단어 빈도 처리 
word_list = []
for sent in tokenized_X_train:
    for word in sent:
      word_list.append(word)

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))

# %%
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
print(vocab[:10])

print('등장 빈도수 하위 10개 단어')
print(vocab[-10:])

# %%
threshold  = 3
total_cnt  = len(word_counts) # 단어의 수
rare_cnt   = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq  = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# %%
# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print('단어 집합의 크기 :', len(vocab))

# %%
## 단어사전 => 단어:정수값 
word_to_index = {}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

# %%
for index, word in enumerate(vocab) :
  word_to_index[word] = index + 2

# %%
vocab_size = len(word_to_index)
print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)

# %%
print('단어 <PAD>와 맵핑되는 정수 :', word_to_index['<PAD>'])
print('단어 <UNK>와 맵핑되는 정수 :', word_to_index['<UNK>'])

# %% [markdown]
# ### 정수 인코딩 

# %%
## 문장 단위 추출 후 단어들을 정수로 인코딩 진행 함수 
def texts_to_sequences(tokenized_X_data, word_to_index):
  encoded_X_data = []
  for sent in tokenized_X_data:
    index_sequences = []
    for word in sent:
      # index_sequences.append(word_to_index.get(word, '<UNK>'))
      try:
          index_sequences.append(word_to_index[word])
      except KeyError:
          index_sequences.append(word_to_index['<UNK>'])
    encoded_X_data.append(index_sequences)
  return encoded_X_data

# %%
# 1. 토큰화 (변수명 통일: train / val / test)
tokenized_X_train = tokenize(X_train)
tokenized_X_val = tokenize(X_val)
tokenized_X_test = tokenize(X_test)

# 2. 정수 인코딩
encoded_X_train = texts_to_sequences(tokenized_X_train, word_to_index)
encoded_X_val = texts_to_sequences(tokenized_X_val, word_to_index)
encoded_X_test = texts_to_sequences(tokenized_X_test, word_to_index)


# %%
# 상위 샘플 2개 출력
for sent in encoded_X_train[:2]:
  print(sent)

# %%
## 정수 => 단어 변환 사전 ( 예 : 영한사전, 기계어자연어사전 )
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

# %%
decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
print('기존의 첫번째 샘플 :', tokenized_X_train[0])
print('복원된 첫번째 샘플 :', decoded_sample)

# %% [markdown]
# ### 패딩

# %%
print('리뷰의 최대 길이 :',max(len(review) for review in encoded_X_train))
print('리뷰의 평균 길이 :',sum(map(len, encoded_X_train))/len(encoded_X_train))
plt.hist([len(review) for review in encoded_X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# %%
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

# %%
max_len = 500
below_threshold_len(max_len, encoded_X_train)

# %%
def pad_sequences(sentences, max_len):
  features = np.zeros((len(sentences), max_len), dtype=int)
  for index, sentence in enumerate(sentences):
    if len(sentence) != 0:
      features[index, :len(sentence)] = np.array(sentence)[:max_len]
  return features

# %%
#pip install tensorflow

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 최대 길이 설정 (예시: 그래프에서 본 최대값 14)
max_len = 14  # 또는 평균+표준편차 고려해서 설정

# 패딩 적용 (변수명 일치시켜야 함!)
padded_X_train = pad_sequences(encoded_X_train, maxlen=max_len, padding='post', truncating='post')
padded_X_val   = pad_sequences(encoded_X_val, maxlen=max_len, padding='post', truncating='post')
padded_X_test  = pad_sequences(encoded_X_test, maxlen=max_len, padding='post', truncating='post')


# %%
print('훈련 데이터의 크기 :', padded_X_train.shape)
print('검증 데이터의 크기 :', padded_X_val.shape) 
print('테스트 데이터의 크기 :', padded_X_test.shape)

# %%
print(padded_X_train.shape[:2])

# %% [markdown]
# ### 모델링

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
train_label_tensor = torch.tensor(np.array(y_train))
val_label_tensor   = torch.tensor(np.array(y_val))
test_label_tensor  = torch.tensor(np.array(y_test))

# %%
train_label_tensor[:5]

# %%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", device)

# %%
# input.shape == (배치 크기, 임베딩 벡터의 차원, 문장 길이)
input = torch.randn(32, 16, 50)

# 선언 시 nn.Conv1d(임베딩 벡터의 차원, 커널의 개수, 커널 사이즈)
m = nn.Conv1d(16, 33, 3, stride=1)

# output.shape == (배치 크기, 커널의 개수, 컨볼루션 연산 결과 벡터)
output = m(input)
print(output.shape)

# %%
# 1. 실제 입력 (예: padded_X_train은 (batch, seq_len) 모양의 정수 인코딩 텐서)
input = torch.tensor(padded_X_train[:32], dtype=torch.long)  # (32, 50) → 배치 크기 32, 문장 길이 50

# 2. 임베딩 적용
embedding = nn.Embedding(num_embeddings=len(word_to_index), embedding_dim=16, padding_idx=0)
embedded = embedding(input)  # (32, 50, 16)

# 3. Conv1d를 위해 (배치, 채널, 시퀀스 길이)로 차원 변경
embedded = embedded.permute(0, 2, 1)  # (32, 16, 50)

# 4. Conv1d 선언 및 적용
m = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=1)
output = m(embedded)

# 5. 출력 확인
print("Conv1d 결과 shape:", output.shape)  # (32, 33, 48)


# %%
class CNN(torch.nn.Module):
  def __init__(self, vocab_size, num_labels):
    super(CNN, self).__init__()

    # 오직 하나의 종류의 필터만 사용함.
    self.num_filter_sizes = 1 # 윈도우 5짜리 1개만 사용
    self.num_filters = 256

    self.word_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)
    # 윈도우 5짜리 1개만 사용
    self.conv1 = torch.nn.Conv1d(128, self.num_filters, 5, stride=1)
    self.dropout = torch.nn.Dropout(0.5)
    self.fc1 = torch.nn.Linear(1 * self.num_filters, num_labels, bias=True)

  def forward(self, inputs):
    # word_embed(inputs).shape == (배치 크기, 문장길이, 임베딩 벡터의 차원)
    # word_embed(inputs).permute(0, 2, 1).shape == (배치 크기, 임베딩 벡터의 차원, 문장 길이)
    embedded = self.word_embed(inputs).permute(0, 2, 1)

    # max를 이용한 maxpooling
    # conv1(embedded).shape == (배치 크기, 커널 개수, 컨볼루션 연산 결과) == ex) 32, 256, 496
    # conv1(embedded).permute(0, 2, 1).shape == (배치 크기, 컨볼루션 연산 결과, 커널 개수)
    # conv1(embedded).permute(0, 2, 1).max(1)[0]).shape == (배치 크기, 커널 개수)
    x = F.relu(self.conv1(embedded).permute(0, 2, 1).max(1)[0])

    # y_pred.shape == (배치 크기, 분류할 카테고리의 수)
    y_pred = self.fc1(self.dropout(x))

    return y_pred

# %%

# 변수 이름 통일
padded_X_valid = padded_X_val
valid_label_tensor = torch.tensor(y_val)

# Tensor 변환 및 Dataset 구성
encoded_train = torch.tensor(padded_X_train).to(torch.int64)
train_label_tensor = torch.tensor(y_train)
train_dataset = torch.utils.data.TensorDataset(encoded_train, train_label_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)

encoded_test = torch.tensor(padded_X_test).to(torch.int64)
test_label_tensor = torch.tensor(y_test)
test_dataset = torch.utils.data.TensorDataset(encoded_test, test_label_tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1)

encoded_valid = torch.tensor(padded_X_valid).to(torch.int64)
valid_dataset = torch.utils.data.TensorDataset(encoded_valid, valid_label_tensor)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=1)


# %%
num_epochs = 5
total_batch = len(train_dataloader)
print('총 배치의 수 : {}'.format(total_batch))

# %%
model = CNN(vocab_size, num_labels = len(set(y_train)))
model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
def calculate_accuracy(logits, labels):
    # _, predicted = torch.max(logits, 1)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# %%
def evaluate(model, valid_dataloader, criterion, device):
    val_loss = 0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        # 데이터로더로부터 배치 크기만큼의 데이터를 연속으로 로드
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 모델의 예측값
            logits = model(batch_X)

            # 손실을 계산
            loss = criterion(logits, batch_y)

            # 정확도와 손실을 계산함
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy

# %%
# Training loop
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    # Training
    train_loss = 0
    train_correct = 0
    train_total = 0
    model.train()
    for batch_X, batch_y in train_dataloader:
        # Forward pass
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        # batch_X.shape == (batch_size, max_len)
        logits = model(batch_X)

        # Compute loss
        loss = criterion(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        train_loss += loss.item()
        train_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_dataloader)

    # Validation
    val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # 검증 손실이 최소일 때 체크포인트 저장
    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')

# %%
# 모델 로드
model.load_state_dict(torch.load('best_model_checkpoint.pth'))

# 모델을 device에 올립니다.
model.to(device)

# 검증 데이터에 대한 정확도와 손실 계산
val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

# %%
def predict(text, model, word_to_index, index_to_tag):
    # 모델 평가 모드
    model.eval()

    # 토큰화 및 정수 인코딩. OOV 문제 발생 시 <UNK> 토큰에 해당하는 인덱스 1 할당
    tokens = word_tokenize(text)
    token_indices = [word_to_index.get(token.lower(), 1) for token in tokens]

    # 리스트를 텐서로 변경
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)  # (1, seq_length)

    # 모델의 예측
    with torch.no_grad():
        logits = model(input_tensor)  # (1, output_dim)

    # 레이블 인덱스 예측
    _, predicted_index = torch.max(logits, dim=1)  # (1,)

    # 인덱스와 매칭되는 카테고리 문자열로 변경
    predicted_tag = index_to_tag[predicted_index.item()]

    return predicted_tag
