# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# Função para gerar números aleatórios para x e y
def gerar_numeros_aleatorios(n):
    x = np.random.uniform(1000, 60000, n)  # Gera n números aleatórios no intervalo [1000, 60000)
    a = 1e-4  # Valor pequeno para 'a' para evitar valores muito grandes em y
    b = np.random.uniform(1, 10)  # Valor aleatório para 'b' no intervalo [1, 10)

    y = (1 / (a * x)) + b  # Calcula y de acordo com a fórmula
    y = np.clip(y, 0, 20)  # Garante que y esteja no intervalo [0, 20]

    return x, y


# Gerar dados x e y
x, y = gerar_numeros_aleatorios(10000)

# Filtrando dados conforme os critérios especificados no primeiro código
x = x[(x >= 1000) & (x <= 60000)]
y = y[y <= 20]


# Dado f((1 'sobre' ax) + b)
def grad(a, b, x, y):
    grad_a = -sum((y[i] - (1 / (a * x[i]) + b)) * (1 / (a**2 * x[i]) if x[i] != 0 else 1e-10) for i in range(len(x)))
    grad_b = -sum(y[i] - (1 / (a * x[i]) + b) for i in range(len(x)))
    return [grad_a, grad_b]


def dist(anterior, novo):  # Calcula a distância euclidiana.
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(anterior, novo)))


def grad_desc(lr, a, b, tol, x, y):
    d = float('inf')  # Inicializa com um valor grande para garantir que o loop execute pelo menos uma vez
    k = 0  # Contador

    while d > tol and k < 1000:  # Limita o número de iterações para evitar loops infinitos
        grads = grad(a, b, x, y)
        a_novo = a - lr * grads[0]
        b_novo = b - lr * grads[1]

        d = dist([a, b], [a_novo, b_novo])
        a, b = a_novo, b_novo
        k += 1
        print(f"Iteração {k}: d = {d}, a = {a}, b = {b}")

    return [a, b, k]


# Parâmetros iniciais e taxa de aprendizado
a_inicial, b_inicial = 1e-4, 4.3
lr = 1e-15
tol = 1e-7

a, b, t = grad_desc(lr, a_inicial, b_inicial, tol, x, y)
print(f'a = {a}\nb = {b}')

# Criação do gráfico
plt.scatter(x, y, color='blue', label='Números aleatórios')  # Plota os Números aleatórios

# Gera pontos para a reta de regressão
x_reta = np.linspace(min(x), max(x), 10000)
y_reta = [(1 / (a * xi) + b) for xi in x_reta]

plt.plot(x_reta, y_reta, color='red', label='Reta de Regressão')  # Plota a reta

plt.xlabel('x')
plt.ylabel('y')
plt.title('Reta de Regressão Linear')
plt.show()