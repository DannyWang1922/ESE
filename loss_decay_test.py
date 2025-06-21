import math

print("\n")
print("ESE shallow loss decay")
ese_list = []
for i in range(11):
	division = (1. + math.log(1 + i))
	weight = 1/division
	weight = round(weight, 4)
	ese_list.append(weight)
print(ese_list)
print("\n")

print("\n")
print("Sentence-Transformer loss decay")
st_list = []
for i in range(11):
	division = (1 + i) / 11
	weight = 1/division
	weight = round(weight, 4)
	st_list.append(weight)
print(st_list)
print("\n")

print("\n")
print("Sentence-Transformer loss decay with prior_layers_weight")
st_list = []
prior_layers_weight = 0.8
for i in range(11):
	division = (1 + i) / 11
	weight = 1/division
	weight = weight * prior_layers_weight
	weight = round(weight, 4)
	st_list.append(weight)
print(st_list)
print("\n")