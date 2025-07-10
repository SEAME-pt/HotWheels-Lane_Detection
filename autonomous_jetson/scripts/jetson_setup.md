# Configuração de Desenvolvimento Remoto - Jetson Nano

## 1. Configurar SSH na Jetson Nano

```bash
# Na Jetson Nano
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh

# Verificar IP da Jetson
ip addr show
```

## 2. Configurar SSH no seu computador

```bash
# Gerar chave SSH (se não tiver)
ssh-keygen -t rsa -b 4096

# Copiar chave para Jetson
ssh-copy-id usuario@IP_DA_JETSON

# Testar conexão
ssh usuario@IP_DA_JETSON
```

## 3. VS Code Remote Development

1. Instalar extensão "Remote - SSH" no VS Code
2. Pressionar Ctrl+Shift+P
3. Digitar "Remote-SSH: Connect to Host"
4. Adicionar: `usuario@IP_DA_JETSON`
5. Abrir pasta do projeto remotamente

## 4. Compilação Remota

```bash
# Via SSH direto
ssh usuario@IP_DA_JETSON "cd /caminho/do/projeto && make"

# Ou usando script
./compile_remote.sh
```

## 5. Sincronização de Arquivos (Alternativa)

```bash
# Usando rsync
rsync -avz --exclude='.git' /local/projeto/ usuario@IP_DA_JETSON:/remote/projeto/
```
