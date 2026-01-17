import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.array(x**2 + 2)

def derivative_f(x):
    return np.array(2*x)

# Данные, по которым строим график
x = np.linspace(-4, 4, 100)
y = f(x)

# Подписи полей
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.title('Градиентный спуск для f(x) = x² + 2')
# Ограничение области просмотра графика
plt.xlim(-5, 5)
plt.ylim(0, 15)
# Сетка на графике
plt.grid(True, alpha=0.3)

# Функция f(x)
plt.plot(x,y, label='f(x) = x² + 2')

# Точка, от которой будем запускать градиентный спуск
point = [x[20], y[20]]
plt.scatter(*point, color='red', alpha=1, label='Начальная точка')

# Функция производной f(x)
plt.plot(x,derivative_f(x), label='Функция производной')

# Градиентный спуск
x_old = point[0]
speed = 0.2 # Шаг обучения
for i in range(20):
    x_new = x_old - speed * derivative_f(x_old)
    plt.scatter(x_new, f(x_new), color='orange')
    x_old = x_new
plt.scatter(x_old, f(x_old), color='red') # Точка минимума

# Уравнение касательной: y = f′(x0)(x − x0) + f(x0)
y = derivative_f(point[0])*(x-point[0]) + f(point[0])
plt.plot(x, y, linestyle='--', label='Касательная')

plt.legend(loc='best')
plt.tight_layout()
plt.show()