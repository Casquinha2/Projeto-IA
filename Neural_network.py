import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from twitchio.ext import commands

#Inserir o tocket client do canal da twitch
token_client = 'AQUI'

#Inserir o nome do/dos canal/canais da twitch
canal = ['AQUI']

#Leitura da base de dados
df = pd.read_csv('ham_spam_xqcow_and_sodapoppin.csv', sep='\t', header=None, names=['label', 'message'])
df[['label', 'message']] = df['label'].str.split(n=1, expand=True)
df['message'] = df['message'].astype(str).str.strip('"')
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

#Definir as variáveis
X = df['message']
y = df['label_encoded']

#Divisão da base de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Vetorização das variáveis X
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#Reduzir dimensão usando TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_train_reduced = svd.fit_transform(X_train_counts)
X_test_reduced = svd.transform(X_test_counts)

#print de teste para sabermos onde estamos
print('Dados prontos para conversão')

#Converter dadps para PyTorch tensors
X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#print de teste para sabermos onde estamos
print('Dados prontos para treino')

#Criação de TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#print de teste para sabermos onde estamos
print('Datasets com os dados para treino')

#Definição do modelo de classificação
class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Criação do modelo de classificação
model = SpamClassifier()

#print de teste para sabermos onde estamos
print('Modelo Criado')

#Definir loss e optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Treinar o modelo com os dados de treinamento
epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


#Treinar o modelo com os dados de treinamento
print('Modelo Treinado')

#Avaliar o modelo
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())

#Mostrar tabela com estatísticas
print(classification_report(y_test, all_preds))
cm = confusion_matrix(y_test, all_preds)
cm = confusion_matrix(y_test, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

#Comandos para o Bot da Twitch
class Bot(commands.Bot):

    def __init__(self):
        super().__init__(token=token_client, prefix='!', initial_channels=canal)

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')

    #Verifica todas as mensagens enviadas no chat e bane o usuário caso seja detectado spam
    async def event_message(self, message):
        if message.echo:
            return
     
        #Tratamento da mensagem
        message_counts = vectorizer.transform([message.content])
        message_reduced = svd.transform(message_counts)
        message_tensor = torch.tensor(message_reduced, dtype=torch.float32)

        #Previção da mensagem
        model.eval()
        with torch.no_grad():
            outputs = model(message_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred_binary = y_pred.item()

        #Detetor da spam e banimento do usuário
        if y_pred_binary == 1:
            await message.channel.send('Spam detectado! Por favor, evite enviar spam.')
            try:
                await message.author.ban(reason='Envio de spam')
                await message.channel.send(f'O usuário {message.author.name} foi banido por spam.')
            except Exception as e:
                await message.channel.send('Não foi possível banir o usuário. Verifique as permissões do bot.')
        else:
            print(f'A mensagem "{message.content}" foi classificada como não spam.')

        await self.handle_commands(message)

#Tenta correr o código para o Bot da Twitch
try:
    bot = Bot()
    bot.run()
except Exception as e:
    print(f"""\nO programa não dá para usar com a conta da twitch.
Erros possíveis:
    - Não querer fazer a interligação com a Twitch
    - Token mal inserido
    - Canal inexistente
Erro: {e}""")
