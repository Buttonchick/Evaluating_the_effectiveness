import numpy as np
import matplotlib.pyplot as plt
import math
from math import asin, sqrt
import random
from scipy.stats import norm, lognorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad


m_mu = 3.2 # математическое ожидание массы частицы
m_sigma = 1.6  # стандартное отклонение массы частицы


# Начальное распределение зарядов
q_mu =-3.6599877196493527
q_sigma = 5.800044109138212
q_A = 0.1314373943594196

# задаем начальные условия
e = -1.6e-19 # Кл
P_atm = 101325 # Па


ms = np.logspace(-3*m_sigma, 3*m_sigma, 1000)  # нг, вероятные значения массы
qs = np.round(np.linspace(q_mu - 3 * q_sigma, q_mu + 3 * q_sigma, int(6 * abs(q_sigma)))) # е, вероятные значения зарядов

aerosols = np.array([(m, q) for m in ms for q in qs ])


filters_params = [
  
    # Каждый элемент в списке `filters_params` содержит следующие значения (в порядке указания):
    #   - Название фильтра
    #   - количесвтво слоёв (в шт)
    #   - Диаметр фильтра (в м)
    #   - Толщина фильтра (в м)
    #   - Средний размер ячейки (в м)
    #   - Напряженность поля (в В/м)
    #   - Эффективность фильтрации (%)
    #   - Сопротивление воздуха (в Па)
  
    ("3M N95", 3, 100e-3, 0.1e-3, 0.25e-6, 8e3, 95 , 15),
    ("Honeywell", 3, 80e-3, 0.5e-3, 0.15e-6, 5e3, 98, 20),
    ("Vogmask", 3, 90e-3, 0.3e-3, 0.15e-6, 7e3, 99, 18),
    ("Respro Techno Plus", 3, 70e-3,0.4e-3, 0.2e-6, 9e3, 99, 25)
]




def probability(m, q):
    """
    Рассчитывает вероятность события на основе логнормального распределения массы частицы и гауссовского распределения заряда.

    Аргументы:
    m -- масса частицы
    q-- заряда частицы

    Возвращает:
    Вероятность события
      """
    # расчет вероятности события 
    prob = lognorm.pdf(m, m_sigma, scale=np.exp(m_mu)) * q_A* np.exp(-(q-q_mu)**2/(2*q_sigma**2))

        
    return np.clip(100*prob, 0.03, 99.994)



def particle_t(q, m, E, p_d, d, v):
    """
    Рассчитывает время подъема заряженной частицы в поле.

    Аргументы:
    q -- заряд частицы (Кл).
    m -- масса частицы (кг).
    E -- напряженность электрического поля (В/м).
    p_d -- диаметр частицы (нм).
    v -- скорость потока воздуха (м/с).

    Возвращает:
    Время подъема заряженной частицы в поле (с).
    """
    # вычисление констант
    x0 = d
    A = (p_d/2)**2 * np.pi
    C1 = (E**2)*A/(2*m)
    C2 = abs(q)*E/(2*m)

    # вычисление времени и глубины оседания
    t = abs((x0 - C2**2)/ C1)** (1/3)
    z = t*v
  
  
    return t,z


def calculate_diameter(m, rho):
    d = 2*(3 * m /(4*math.pi * rho)) ** (1/3)
    return d



electrode_params = []

for filter_params in filters_params:
    name, n, D, L, d, E, nominal_eff, dP = filter_params   # Распаковка значений фильтра
  
    # Площадь поперечного сечения трубы
    A = math.pi * (D/2) ** 2
  
    v = 0.001/A 
  
    V2 = v*(P_atm-dP)/P_atm

    
    # Электрическая постоянная поля
          
    lost_aerosols = []
    cought_aerosols = []
    
    # Итерация аэрозолей
    aerosols = np.array(aerosols)
    m = aerosols[:, 0]
    q = aerosols[:, 1]

    p_d = calculate_diameter(m * 1e-12, 1000)
    
    _, z = particle_t(q * e, m * 1e-12, E, p_d, d, V2)
    probs = probability(m, q)
    
    cought_mask = z <= L*n
    cought_aerosols = aerosols[cought_mask]
    lost_aerosols = aerosols[~cought_mask]
    
    cought_prob = np.sum(probs[cought_mask])
    lost_prob = np.sum(probs[~cought_mask])
    eff = cought_prob / (cought_prob + lost_prob)

    print(eff)
  
    
    electrode_params.append((name, 100*eff, nominal_eff))





# Полученные и номинальные значения эффективности фильтрации
efficiency_obtained = [item[1] for item in electrode_params]
efficiency_nominal = [item[2] for item in electrode_params]

# Названия фильтров
filter_names = [item[0] for item in electrode_params]

# Создание графика
fig, ax = plt.subplots()

# Построение столбчатых диаграмм для полученной и номинальной эффективности
bar_width = 0.35
opacity = 0.8
index = np.arange(len(electrode_params))

rects1 = ax.bar(index, efficiency_obtained, bar_width,
                alpha=opacity, color='b',
                label='Полученная эффективность')

rects2 = ax.bar(index + bar_width, efficiency_nominal, bar_width,
                alpha=opacity, color='r',
                label='Номинальная эффективность')

# Настройка осей и заголовка
ax.set_xlabel('Фильтры')
ax.set_ylabel('Эффективность фильтрации (%)')
ax.set_title('Сравнение полученной и номинальной эффективности фильтрации')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(filter_names)
ax.legend()

# Отображение графика
plt.tight_layout()
plt.show()
