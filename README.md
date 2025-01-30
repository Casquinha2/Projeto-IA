# Projeto-IA
 Detetor de spam para banimentos na plataforma Twitch (projeto para a cadeira de Inteligência Artificial)

# Instruções de uso deste programa
 Este código dá para ser usado sem a interligação com a Twitch, mas nesse caso o único output esperado é só as estatisticas de cada método de machine learning.
 
 Se pretender usar o programa o canal da Twitch, por favor insira o Token do cliente no começo do código para o programa ter acesso ao seu chat. 
 
 Para ter um token para este programa basta vir a este site https://twitchtokengenerator.com, logar a conta da sua Twitch e ativar os seguintes Scopes:
 
     -chat:read    
     
     -chat:edit  
     
     -channel:moderate
     
   
Após a ativação destes Scopes, é só gerar um novo Token, copiar o "ACCESS TOKEN" gerado e meter no início do código.

Após a inserção do Token, é necessário meter o nome do canal (exemplo: casquinha2).

Ao finalizar estes passos é só dar run ao programa e esperar que a mensagem de "Logged in as ()" e "User id is ()" apareça no terminal.

Agora já está pronto para usar este programa para a deteção e banimento de usuários.

Lembrando que só necessita de escolher um dos ficheiros que são:

     -Naives_Bayes.py
     
     -Linear_Regression.py
     
     -Neural_Network.py
