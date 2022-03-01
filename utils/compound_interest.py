btc = 0.2

# A = P(1 + r/n)^nt
r = 0.05
n = 365
r_div_n = r / n
t = 7 / n # 1 week
year = 52
#t = 30 / n # 1 month
#year = 12

for i in range(year):
    btc *= (1 + r_div_n)**(n*t)
    btc += 0.008

print(btc)
