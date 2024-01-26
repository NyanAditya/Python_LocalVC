money = int(input('Enter Amount: '))

if money >= 1000000000000000000000000000000:
	print('Over the Limit')

else:
	TH = money // 2000
	FiveH = (money % 2000) // 500
	TwoH = ((money % 2000) % 500) // 200
	H = ((((money % 2000) % 500) % 200) % 100) // 100
	Ft = ((((money % 2000) % 500) % 200) % 100) // 50
	Tw = (((((money % 2000) % 500) % 200) % 100) % 50) // 20
	Te = ((((((money % 2000) % 500) % 200) % 100) % 50) % 20) // 10
	Fi = (((((((money % 2000) % 500) % 200) % 100) % 50) % 20) % 10) // 5
	To = ((((((((money % 2000) % 500) % 200) % 100) % 50) % 20) % 10) % 5) // 2
	on = (((((((((money % 2000) % 500) % 200) % 100) % 50) % 20) % 10) % 5) % 2) // 1

	if TH > 0:
		print('Rs. 2000 x ', TH)
	if FiveH > 0:
		print('Rs. 500 x ', FiveH)
	if TwoH > 0:
		print('Rs. 200 x ', TwoH)
	if H > 0:
		print('Rs. 100 x ', H)
	if Ft > 0:
		print('Rs. 50 x ', Ft)
	if Tw > 0:
		print('Rs. 20 x ', Tw)
	if Te > 0:
		print('Rs. 10 x ', Te)
	if Fi > 0:
		print('Rs. 5 x ', Fi)
	if To > 0:
		print('Rs. 2 x ', To)
	if on > 0:
		print('Rs. 1 x ', on)
