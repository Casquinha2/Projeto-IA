import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
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

#Inicializar o classificador Linear Regression
clf = LinearRegression()

#print de teste para sabermos onde estamos
print('Classificador criado')

#Treinar o modelo com os dados de treinamento
clf.fit(X_train_counts, y_train)

#print de teste para sabermos onde estamos
print ("Dados treinados")

#Previsão dos Y
y_pred = clf.predict(X_test_counts)

#Precisão do método
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

#Definir se os dados vão ser 1 ou 0
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

#Mostrar tabela com estatísticas
print(classification_report(y_test, y_pred_binary, target_names=['Ham', 'Spam']))
cm = confusion_matrix(y_test, y_pred_binary)
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
        message_counts = vectorizer.transform(message.content)
        message_reduced = svd.transform(message_counts)

        #Previção da mensagem
        y_pred = clf.predict(message_reduced)
        
        #Binariza a predição para classificar como spam ou não spam
        y_pred_binary = 1 if y_pred >= 0.5 else 0

        #Detetor da spam e banimento do usuário
        if y_pred_binary == 1:
            await message.channel.send('Spam detectado! Por favor, evite enviar spam.')
            try:
                await message.author.ban(reason='Envio de spam')
                await message.channel.send(f'O usuário {message.author.name} foi banido por spam.')
            except Exception as e:
                await message.channel.send('Não foi possível banir o usuário. Verifique as permissões do bot.')

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
