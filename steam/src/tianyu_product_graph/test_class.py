class Complex:
    i = 3
    def __init__(self): 
        self.t = 1
        self.r = 2
    def x(self): 
        self.t += 2
    i *= 2

if __name__ == '__main__': 
    y = Complex()
    print(y.t, y.r, y.i)
    y.x()
    y.i *= 2
    print(y.t, y.r, y.i)
