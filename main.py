from math import fabs, log, log2  # Модлуль числа, логарифми.
from tabulate import tabulate  # Табуляція списку для отримання рядка форматованої таблиці списку.
from sympy import symbols, Eq, real_roots, nsolve # Для обрахунку коренів.
import matplotlib.pyplot as plt  # Малювання графіку залежності.
import numpy as np # Для зберігання й обчислення діпазону значення Х і У функції.

# Обчислення значення функції.
def f(x=float) -> float:
    return 2 ** x - 2 * x ** 2 - 1


# Обчислення значення першої похідної функції.
def f1(x=float) -> float:
    return 2 ** x * log(2) - 4 * x


# Обчислення значення другої похідної функції.
def f2(x=float) -> float:
    return 2 ** x * log(2) ** 2 - 4


# Обчислення значення ітераційної функції.
def phi(x=float) -> float:
    return log2(2 * x ** 2 + 1)


# Обчислення значення похідної ітераційної функції.
def phi1(x=float) -> float:
    return 4 * x / (log(2) * (2 * x ** 2 + 1))


# Метод половинного поділу (дихотомії).
def middleDivisionMethod(a=float, b=float, eps=float, countMaxIteration=int) -> list:
    allListIteration = []
    for i in range(countMaxIteration):
        listIteration = []
        x = (a + b) / 2
        listIteration.append(f"Ітерація # {i + 1}")
        listIteration.append(f"a{i}: {a}")
        listIteration.append(f"b{i}: {b}")
        listIteration.append(f"x{i}: {x}")
        listIteration.append(f"{fabs(f(x)) > eps}")
        if fabs(f(x)) > eps:
            listIteration.append(f"{f(a) * f(x) > 0}")
            if f(a) * f(x) > 0:
                a = x
                listIteration.append(f"a{i} = x{i}")
            else:
                b = x
                listIteration.append(f"b{i} = x{i}")
        else:
            listIteration[-1] += " =>"
            listIteration.append(f"Корінь знайдено => ")
            listIteration.append(f"x = {x}")
            allListIteration.append(listIteration)
            return [i + 1, x, allListIteration]
        allListIteration.append(listIteration)
    else:
        raise Exception(
            f"На цьому проміжку [{a}; {b}] - коренів не знайдено або цей проміжок не підходить для цього методу!")


# Метод хорд.
def chordMethod(a=float, b=float, eps=float, countMaxIteration=int) -> list:
    allListIteration = []
    c = 0.0
    for i in range(countMaxIteration):
        listIteration = []
        listIteration.append(f"Ітерація # {i + 1}")
        if i == 0 and f(a) > 0:
            a, b = b, a
            listIteration.append(f"a{i}: {a} тому що f(a) > 0")
            listIteration.append(f"b{i}: {b} тому що f(a) > 0")
        else:
            listIteration.append(f"a{i}: {a}")
            listIteration.append(f"b{i}: {b}")
        listIteration.append(f"{fabs(b - a) > eps}")
        if fabs(b - a) > eps:
            if (f(b) - f(a)) == 0:
                raise ZeroDivisionError("Ділення на нуль!")
            c = a - (b - a) / (f(b) - f(a)) * f(a)
            listIteration.append(f"c{i}: {c}")
            listIteration.append(f"{f(a) * f(c) > 0}")
            if f(a) * f(c) > 0.0:
                a = c
                listIteration.append(f"a{i} = c{i}")
            else:
                b = c
                listIteration.append(f"b{i} = c{i}")
        else:
            listIteration[-1] += " =>"
            listIteration.append(f"Корінь знайдено => ")
            listIteration.append(f"x = {c}")
            listIteration.append("-" * 10)
            allListIteration.append(listIteration)
            return [i + 1, a, allListIteration]
        allListIteration.append(listIteration)
    else:
        raise Exception(
            f"На цьому проміжку [{a}; {b}] - коренів не знайдено або цей проміжок не підходить для цього методу!")


# Метод дотичних.
def secantMethod(a=float, b=float, eps=float, countMaxIteration=int) -> list:
    x0 = 0.0
    allListIteration = []
    for i in range(countMaxIteration):
        listIteration = []
        listIteration.append(f"Ітерація # {i + 1}")
        if i == 0:
            if f(a) * f2(a) > 0.0:
                x0 = a
                listIteration.append(f"x{i}: {x0} (x0 = a тому що f(a) * f2(a) > 0.0)")
            else:
                x0 = b
                listIteration.append(f"x{i}: {x0} (x0 = b тому що f(a) * f2(a) <= 0.0)")
        else:
            listIteration.append(f"x{i}: {x0}")
        if (f1(x0) == 0.0):
            raise ZeroDivisionError("Ділення на нуль!")
        xr = x0 - f(x0) / f1(x0)
        listIteration.append(f"xr{i}: {xr}")
        listIteration.append(f"{fabs(x0 - xr) > eps}")
        if fabs(x0 - xr) > eps:
            x0 = xr
            listIteration.append(f"x = xr")
        else:
            listIteration[-1] += " => Корінь знайдено => "
            listIteration.append(f"x = {xr}")
            allListIteration.append(listIteration)
            return [i + 1, xr, allListIteration]
        allListIteration.append(listIteration)
    else:
        raise Exception(
            f"На цьому проміжку [{a}; {b}] - коренів не знайдено або цей проміжок не підходить для цього методу!")


# Метод простої ітерації.
def simpleIterationMethod(a=float, b=float, eps=float, countMaxIteration=int) -> list:
    x0 = 0.0
    allListIteration = []
    for i in range(countMaxIteration):
        listIteration = []
        listIteration.append(f"Ітерація # {i + 1}")
        if i == 0:
            if not (fabs(phi1(a)) < 1 and fabs(phi1(b)) < 1):
                raise Exception(
                    f"На цьому проміжку [{a}; {b}] - коренів не знайдено або цей проміжок не підходить для цього методу!")
            else:
                x0 = a
                listIteration.append(f"x{i}: {x0} тому що phi'(a) i phi'(b) < 1")
        else:
            listIteration.append(f"x{i}: {x0}")
        xr = phi(x0)
        listIteration.append(f"xr{i}: {xr}")
        listIteration.append(f"{fabs(xr - x0) > eps}")
        if fabs(xr - x0) > eps:
            x0 = xr
            listIteration.append(f"x0 = xr")
        else:
            listIteration[-1] += " => Корінь знайдено => "
            listIteration.append(f"x = {xr}")
            allListIteration.append(listIteration)
            return [i + 1, xr, allListIteration]
        allListIteration.append(listIteration)
    else:
        raise Exception(
            f"На цьому проміжку [{a}; {b}] - коренів не знайдено або цей проміжок не підходить для цього методу!")


# Рядок (таблиця) Методу.
def strMethodInTable(func, a=float, b=float, eps=float, countMaxIteration=int, nameMethod=str,
                     headersTable=list) -> str:
    nameTable = f"ЛР № 4, варіант № 3, Вальчевський П. В., ОІ-11 сп - Знаходження кореня за допомогою {nameMethod}:"
    descriptionTable = f"\t* взято відрізок [{a}; {b}], епсолон: {eps}, максимальна кількість ітерацій: {countMaxIteration}."
    table = tabulate(func(a, b, eps, countMaxIteration)[2], headersTable, tablefmt="pretty")
    return f"{nameTable}\n{descriptionTable}\n{table}"


# Отримати дані результату виконання Методу.
def getDataOfResultMethod(func, nameMethod=str, a=float, b=float, eps=float, countMaxIteration=int) -> list:
    funcResult = func(a, b, eps, countMaxIteration)
    return [nameMethod, f"{funcResult[0]}", f"{funcResult[1]}"]


# Рядок (таблиця) результатів виконання програми.
def strResultInTable(a=float, b=float, eps=float, countMaxIteration=int) -> str:
    nameTable = f"ЛР № 4, варіант № 3, Вальчевський П. В., ОІ-11 сп - Результати програми:"
    descriptionTable = f"\t* взято відрізок [{a}; {b}], епсолон: {eps}, максимальна кількість ітерацій: {countMaxIteration}."
    listResult = [
        getDataOfResultMethod(middleDivisionMethod, "Метод половинного поділу", a, b, eps, countMaxIteration),
        getDataOfResultMethod(chordMethod, "Метод хорд", a, b, eps, countMaxIteration),
        getDataOfResultMethod(secantMethod, "Метод дотичних", a, b, eps, countMaxIteration),
        getDataOfResultMethod(simpleIterationMethod, "Метод простої ітерації", a, b, eps, countMaxIteration)
    ]
    table = tabulate(listResult, ["Назва методу", "Кількість ітерацій", "Корінь"], tablefmt="pretty")
    return f"{nameTable}\n{descriptionTable}\n{table}"


# Демонстрація графіку залежності кількості ітерацій до епсолону.
def showGraphicMethodIterationAndEps(func, nameFunc, a=float, b=float, countMaxIteration=int) -> None:
    powEps = [-i for i in range(1, 10 + 1)]
    countIterationInPowEps = [func(a, b, 10 ** i, countMaxIteration)[0] for i in powEps]
    plt.plot(powEps, countIterationInPowEps, label='Залежність')
    plt.xlabel("Степінь епсолона в форматі 10 ^ i")
    plt.ylabel("Кікільсть ітерацій")
    plt.title(f"ЛР № 4, варіант № 3, Вальчевський П. В., ОІ-11 сп\nГрафік {nameFunc}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Вивід графіку функції.
def drawFx(a=int, b=int) -> None:
    # Константи підібрані вручну.
    num = 40 # Кількість Х для подальшого зображення графіку.
    aY = -40 # Нижня межа У.
    bY = 40 # Верхня межа У.
    # Діапазон Х.
    X = np.linspace(a, b, num)
    # Обраховуємий діапазон У.
    Y = 2 ** X - 2 * X ** 2 - 1
    # Малювання графіку.
    plt.plot(X, Y, label='Графік функції', color='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('ЛР № 4, Варіант № 3, Вальчевський П. В., група ОІ-11 сп')
    plt.legend()
    plt.grid(True)
    plt.xlim(a, b)
    plt.ylim(aY, bY)
    plt.axvline(0, color='r', linewidth=1)  # Вісь X.
    plt.axhline(0, color='r', linewidth=1)  # Вісь Y.
    plt.show()

# Обрахунок й вивід коренів функції.
def calcAndOutputRoots() -> None:
    X = symbols('X') # Оголошення символу для подальшого обрахунку функції.
    equation = Eq(2 ** X - 2 * X ** 2 - 1, 0) # Розрахунок функції.
    print("Один з коренів рівння, який шукався у програмі (для перевірки за допомогою сторінніх бібліотек):")
    root = nsolve(equation, X, 6.0)  # Пошук кореня близького до певного значення.
    print(f"\tКорінь рівняння: {round(root, 10)}")


# Програма виконання.
if __name__ == "__main__":
    a, b, eps, countMaxIteration = 5, 8, 10 ** -5, 10 ** 10
    print(strMethodInTable(middleDivisionMethod, a, b, eps, countMaxIteration, "Методу половинного поділу (дихотомії)",
                           ["Ітерація", "a", "b", "x", "Умова: |f(x)| > eps", "Умова: f(a) * f(x) > 0", "Заміна"]))
    print(strMethodInTable(chordMethod, a, b, eps, countMaxIteration, "Методу хорд",
                           ["Ітерація", "a", "b", "Умова: |b - a| > eps", "c", "Умова: f(a) * f(c) > 0", "Заміна"]))
    print(strMethodInTable(secantMethod, a, b, eps, countMaxIteration, "Методу дотичних",
                           ["Ітерація", "x0", "xr", "Умова: |x0 - xr| > eps", "Заміна"]))
    print(strMethodInTable(simpleIterationMethod, a, b, eps, countMaxIteration, "Методу простої ітерації",
                           ["Ітерація", "x0", "xr", "Умова: |x0 - xr| > eps", "Заміна"]))
    print(strResultInTable(a, b, eps, countMaxIteration))
    showGraphicMethodIterationAndEps(middleDivisionMethod, "Методу половинного поділу (дихотомії)", a, b,
                                     countMaxIteration)
    showGraphicMethodIterationAndEps(chordMethod, "Методу хорд", a, b, countMaxIteration)
    showGraphicMethodIterationAndEps(secantMethod, "Методу дотичних", a, b, countMaxIteration)
    showGraphicMethodIterationAndEps(simpleIterationMethod, "Методу простих ітерацій", a, b, countMaxIteration)
    drawFx(a, b)
    calcAndOutputRoots()
    print("Графіки залежності епсолона й кількості ітерацій в інших вікнах!")
