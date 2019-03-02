import numpy as np
from decimal import Decimal
from uncertainties import ufloat, unumpy, core
import os 
import matplotlib.pyplot as plt

VVp = '0.10003'

print('python -m pip install dist/At_connect-',VVp,'-py3-none-any.whl')

def apetecan():
	return

def n(arr):
    return unumpy.nominal_values(arr)
	
def s(arr):
    return unumpy.std_devs(arr)
	
def fexp(number):
	#(sign, digits, exponent) = Decimal(number).as_tuple()
	#return len(digits) + exponent -1
	return int(np.floor(np.log10(np.abs(number))))

def fman(number):
	number = float(number)
	fman = number/(10**fexp(number))
	return float(fman)
	
def incert(lista_incert,derivadas):
	n = len(lista_incert)
	incert = 0
	for i in range(n):
		incert = incert + derivadas[i]**2*lista_incert[i]**2
	return np.sqrt(incert)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
	
def roundIn5(x,ncs):
	marxe = ncs + 5
	sx = '{:.{}e}'.format(x,marxe)
	if '.' in sx:
		sx = sx.split('.')[0]+sx.split('.')[1]

	Par = False
	Cinco = False
		
	if int(sx[ncs-1])%2==0:
		Par = True
	if sx[ncs] == '5':
		Cinco = True
	Trunca = False
	if Par and Cinco: #if Pan and Circo
		for i in range(ncs+1,len(sx)):
			if sx[i] == 'e':
				Trunca = True
				break
			if sx[i]!= '0':
				break
	if Cinco:
		if Trunca:
			x = x - 10**(fexp(x)-ncs)
		else:
			x = x + 10**(fexp(x)-ncs)
	return x

def turn(arr):
	if len(np.shape(arr)) != 1:
		return arr.T
	x = np.zeros(np.shape(arr))
	for i in np.arange(len(arr)-1,-1,-1):
		print(len(arr) - i)
		x[len(arr) - i-1] = arr[i]
		
	return x	


'''def formato(numero, incertez):
	numero = float(numero)
	if '.' in str(numero) and ('e' not in str(numero)):
		if str(numero).split('.')[1] == '0':
			x = int(numero)
		else:
			x = numero
	else:
		x = numero

	if incertez == '!': # non ten incerteza
		formato = str(x)
	else:
		fx = [fman(x),fexp(x)]
		fsx = [fman(eval(str(incertez))),fexp(eval(str(incertez)))]

		if '.' in str(incertez):
			if int(str(fsx[0]).split('.')[1]) == 0:
				sx = int('%.2g' % int(eval(str(fsx[0]))))
			else:
				sx = float('%.2g' % eval(str(fsx[0])))
		else: 
			sx = int('%.2g' % int(eval(str(fsx[0]))))

		if incertez == '0': #x = incerteza
			if -3<=fx[1]<=1 and len(str(float(('%.2g' % x))).split('.')[1])<=4:
				if (str(float(('%.2g' % x))).split('.')[1] == '0'):
					formato = (int(float(('%.2g') % fx[0])*10**fx[1]))
				else:
					if str(fman(('%.2g') % fx[0])).split('.')[1] == '0' and len(str(fx[0])) > 3:
						formato = str(float(('%.2g') % fx[0])*10**fx[1])+'0'
					else:
						formato = (float(('%.2g') % fx[0])*10**fx[1])
			else:
				formato = ('(%.1f' % fx[0] + ' $ \\cdot \\ 10^{%i})$' % fx[1])
		else:
			if str(fman(('%.2g') % fsx[0])).split('.')[1] == '0' and len(str(fsx[0])) >3:
				aumento = '0'
			else:
				aumento = ''

			if fsx[1] == -3 and len(str(float(incertez)).split('.')[1])<=4:
				formato = (('%.' + str(len(str(float(incertez)).split('.')[1]+aumento)) + 'f') % (fx[0]*10**fx[1]))
			elif (-3<fsx[1]<0):
				formato = (('%.' + str(len(('%.2g' % (incertez)).split('.')[1]+aumento)) + 'f') % (fx[0]*10**fx[1]))
			elif (0<=fsx[1]<=1) and (len(str(float(('%.2g') % fsx[0])).split('.')[0] + str(float(('%.2g') % fsx[0])).split('.')[1])<=3):
				if ((str(float(('%.2g') % fsx[0])*10**fsx[1])).split('.')[1] == '0'):
					if type(x) == int:
						dec = '0'
					else:
						dec = str(abs(fsx[1]-1))
					formato = (('%.' + dec + 'f') % (fx[0]*10**fx[1]))
				else:
					dec = str(abs(fsx[1]-1))
					formato = (('%.' + dec + 'f') % (fx[0]*10**fx[1]))
			else:
				if 'e' not in str(incertez):
					if (('%.2f') % (round(fx[0]*10**(fx[1]-fsx[1]),2))).split('.')[1] == '00' and (str(float(('%.2g') % fsx[0])*10**fsx[1])).split('.')[1] == '0':
						dec = '0'
					else:
						dec = '1'
				else:
					if (('%.2f') % (round(fx[0]*10**(fx[1]-fsx[1]),2))).split('.')[1] == '00' and str(fsx[0]) == '1.0' :
						dec = '0'
					else:
						dec = '1'
				formato = (('(%.'+dec+'f') % (fx[0]*10**(fx[1]-fsx[1])) + ' $ \\cdot \\ 10^{%i})$' % fsx[1])

	return formato'''
	
def auxfix(n,s,L = False, P = False, Ln = False, Ls = False, Pn = True, Ps = True, ncs = 2):
	#ncs = Numero de Cifras Significativas;
	
	aux = fexp(n)-fexp(s) + ncs
	
	n = float('{:.{prec}g}'.format(roundIn5(n,aux), prec = aux))
	s = float('{:.{prec}g}'.format(roundIn5(s,ncs), prec = ncs))
	
	fix = ufloat(n,s)
	fixed = []
	if Pn:
		fixed.append('{:.{}ufL}'.format(fix,ncs).split('\\pm')[0])
	if Ps:
		fixed.append('{:.{}ufL}'.format(fix,ncs).split('\\pm')[1])
	if Ln:
		fixed.append('$ {:.{}ufL}'.format(fix,ncs).split('\\pm')[0]+'$')
	if Ls:
		fixed.append('$'+'{:.{}ufL} $'.format(fix,ncs).split('\\pm')[1])
	if P: 
		fixed.append('{:.{}ufL}'.format(fix,ncs).split('\\pm')[0] + '±' + '{:.{}ufL}'.format(fix,ncs).split('\\pm')[1])
	if L:
		fixed.append('${:.{}ufL}'.format(fix,ncs).split('\\pm')[0] + '\pm' + '{:.{}ufL}$'.format(fix,ncs).split('\\pm')[1])
	return fixed
	
def fix(u, L = False, P = False, Ln = False, Ls = False, Pn = True, Ps = True, ncs = 2):
	sockets = L + P + Ln + Ls + Pn + Ps
	if type(u) == list and len(u) == 2:
		fixed = auxfix(u[0], u[1], L = L, P = P, Ln = Ln, Ls = Ls, Pn = Pn, Ps = Ps, ncs = ncs)
	if type(u) == np.ndarray:
		sh = np.shape(u) 
		if len(sh) == 1:
			u = np.array([u])
			
		fixed = np.zeros((sockets,np.shape(u)[0],np.shape(u)[1]),dtype = object)
		for i in np.arange(np.shape(u)[0]):
			for j in np.arange(np.shape(u)[1]):
				for k in np.arange(sockets):
					fixed[k,i,j] = auxfix(u[i,j].n,u[i,j].s, L = L, P = P, Ln = Ln, Ls = Ls, Pn = Pn, Ps = Ps, ncs = ncs)[k]
		if len(sh) == 1: 
			fixed = fixed[0:sockets,0,:]
	if len(np.shape(u)) == 0 and (type(u) == core.Variable or type(u)== core.AffineScalarFunc):
		fixed = auxfix(u.n,u.s, L = L, P = P, Ln = Ln, Ls = Ls, Pn = Pn, Ps = Ps, ncs = ncs)
	return fixed
	
	
def regresion_s_con_a(x,y,nomex,nomey,nomefigura, pasando = False):
	n = len(y)					# Non te asustes, hai dous tipos de s,b,s(b) e de R con e sen a 1 e 2 respectivmte.
	Sx = sum(x)
	Sy = sum(y)
	Sxx = np.dot(x,x)
	Syy = np.dot(y,y)
	Sxy = np.dot(x,y)
	label = [nomey,nomex]

	a = (Sy*Sxx-Sx*Sxy)/(n*Sxx-Sx**2)
	b = (n*Sxy-Sx*Sy)/(n*Sxx-Sx**2) #b con a

	s = np.sqrt((np.dot(y-a-b*x,y-a-b*x))/(n-2)) #s con a

	sa = s*np.sqrt((Sxx)/(n*Sxx-Sx**2))
	sb = s*np.sqrt((n)/(n*Sxx-Sx**2)) #s(b) con a

	R = (n*Sxy-Sx*Sy)/(np.sqrt((n*Sxx-Sx**2)*(n*Syy-Sy**2))) #R con a

	if pasando:
		f = np.linspace(0,max(x)*(1+0.1),100)
	else:
		f = x

	plt.figure(1); plt.clf()
	plt.plot(f,a + b*f, 'r-')
	plt.plot(x,y, 'b*')

	plt.xlim([min(f)/(1-0.1),max(f)])
	plt.ylim([(min(f)*b+a)/(1-0.1),(max(f)*b+a)])

	#plt.errorbar(x,y,0.017,0.16*0.02,'k')

	plt.xlabel(label[1]);plt.ylabel(label[0])

	sa_red = '%.2g' % sa
	sb_red = '%.2g' % sb
	r_str = str(R); r_red = r_str[0]+r_str[1]; 
	for i in range(2,len(r_str)):
		r_red = r_red + r_str[i]
		if r_str[i] != '9':
			break
	r_sux = str(len(r_red.split('.')[1]))

	print ('Datos Regresion simple con a\n','------------------------------------------------------------------------------------')
	print ('a = ',a,'        s(a) = ',sa)
	print ('b = ',b,'        s(b) = ',sb)
	print ('R = ',R )
	print ('s = ',s)
	print ('SUXERENCIAS:\n'+'sa: [' + sa_red +'] \n'+'sb: [' + sb_red +'] \n'+'R: [' + r_red +']')
	print ('------------------------------------------------------------------------------------')

	fmt_a = fix(a,sa,Ls= False)[0]
	fmt_b = fix(b,sb,Ls = False)[0]

	plt.title(('$y$ $=$'+fmt_a+' $+$ '+fmt_b+'$x$       $R$ = %s') % (r_red))

	plt.grid(False)
	plt.show(False)
	if os.path.isdir('graph') == False:
		os.mkdir('graph')
	plt.savefig('graph/' + nomefigura)
	return(a,sa,b,sb,R)

def regresion_s_sen_a(x,y,nomex,nomey,nomefigura,pasando = False):
	n = len(y)					# Non te asustes, hai dous tipos de s,b,s(b) e de R con e sen a 1 e 2 respectivmte.
	Sx = sum(x)
	Sy = sum(y)
	Sxx = np.dot(x,x)
	Syy = np.dot(y,y)
	Sxy = np.dot(x,y)
	label = [nomey,nomex]

	b = Sxy/Sxx #b sen a

	s = np.sqrt((np.dot(y-b*x,y-b*x)/(n-1)))

	sb = s/np.sqrt(Sxx)

	R = (Sxy)/np.sqrt(Sxx*Syy)

	if pasando:
		f = np.linspace(0,max(x)*(1+0.1),100)
	else:
		f = x
	plt.figure(2); plt.clf()
	plt.plot(f,b*f,'b-')
	plt.plot(x,y, '*', color='orange')

	plt.xlim([min(f)/(1-0.1),max(f)])
	plt.ylim([min(f)*b/(1-0.1),max(f)*b])
	#plt.errorbar(x,y,0.017,0.16*0.02,'k')

	plt.xlabel(label[1]);plt.ylabel(label[0])

	sb_red = '%.2g' % sb
	r_str = str(R); r_red = r_str[0]+r_str[1]; 
	for i in range(2,len(r_str)):
		r_red = r_red + r_str[i]
		if r_str[i] != '9':
			break
	r_sux = str(len(r_red.split('.')[1]))

	fmt_b = fix(b,sb,Ls= False)[0]

	plt.title(('$y$ $=$ '+fmt_b+'$x$       $R$ = %s') % (r_red))

	print ('Datos Regresion simple sen a\n','--------------------------------------------------------')
	print ('b = ',b,'        s(b) = ',sb)
	print ('R = ',R)
	print ('s = ',s)
	print ('SUXERENCIAS:\n'+'sb: [' + sb_red +']\n'+'R: [' + r_red +']')
	print ('--------------------------------------------------------')
	plt.grid(False)
	plt.show(False)
	if os.path.isdir('graph') == False:
		os.mkdir('graph')
	plt.savefig('graph/' + nomefigura)
	return(b,sb,R)

def regresion_p_con_a(x,y,sy,nomex,nomey,nomefigura,pasando = False):
	n = len(y)
	w = 1/(sy**2)
	sw = sum(w)
	swy = sum(w*y)
	swx = sum(w*x)
	swxx = sum(w*x*x)
	swxy = sum(w*x*y)
	d = sw*swxx-swx*swx
	a = (swy*swxx-swx*swxy)/d
	b = (sw*swxy-swx*swy)/d
	sa = np.sqrt(swxx/d)
	sb = np.sqrt(sw/d)
	R = (sw*swxy-swx*swy)/np.sqrt(d*(sw*sum(w*y*y)-swy*swy))

	na = sum(w*(y-a-b*x)**2)
	s = np.sqrt(na*n/((n-2)*sw))

	label = [nomey,nomex]

	if pasando:
		f = np.linspace(0,max(x)*(1+0.1),100)
	else:
		f = x

	plt.figure(1); plt.clf()
	plt.plot(f,a + b*f, 'r-')
	plt.plot(x,y, 'b*')

	plt.xlim([min(f)/(1-0.1),max(f)])
	plt.ylim([(min(f)*b+a)/(1-0.1),(max(f)*b+a)])

	#plt.errorbar(x,y,0.017,0.16*0.02,'k')

	plt.xlabel(label[1]);plt.ylabel(label[0])

	sa_red = '%.2g' % sa
	sb_red = '%.2g' % sb
	r_str = str(R); r_red = r_str[0]+r_str[1]; 
	for i in range(2,len(r_str)):
		r_red = r_red + r_str[i]
		if r_str[i] != '9':
			break
	r_sux = str(len(r_red.split('.')[1]))

	print ('Datos Regresion lineal ponderada con a\n','------------------------------------------------------------------------------------')
	print ('a = ',a,'        s(a) = ',sa)
	print ('b = ',b,'        s(b) = ',sb)
	print ('R = ',R )
	print ('s = ',s)
	print ('SUXERENCIAS:\n'+'sa: [' + sa_red +']\n'+'sb: [' + sb_red +']\n'+'R: [' + r_red +']')
	print ('------------------------------------------------------------------------------------')

	fmt_a = fix(a,sa,Ls= False)[0]
	fmt_b = fix(b,sb,Ls= False)[0]

	plt.title(('$y$ $=$'+fmt_a+' $+$ '+fmt_b+'$x$       $R$ = %s') % (r_red))

	plt.grid(False)
	plt.show(False)
	if os.path.isdir('graph') == False:
		os.mkdir('graph')
	plt.savefig('graph/' + nomefigura)
	return(a,sa,b,sb,R)

def regresion_p_sen_a(x,y,sy,nomex,nomey,nomefigura,pasando = False, text = False, plot = False):
	n = len(y)
	w = 1/(sy**2)
	sw = sum(w)
	swxx = sum(w*x*x)
	swyy = sum(w*y*y)
	swxy = sum(w*x*y)

	b = swxy/swxx
	sb = 1/np.sqrt(swxx)
	R = swxy/np.sqrt(swxx*swyy)

	na = sum(w*(y-b*x)**2)
	s = np.sqrt(na*n/((n-1)*sw))

	label = [nomey,nomex]
	
	if pasando:
		f = np.linspace(0,max(x)*(1+0.1),100)
	else:
		f = x
		
	if plot:
		plt.figure(2); plt.clf()
		plt.plot(f,b*f,'b-')
		plt.plot(x,y, '*', color='orange')

		#plt.xlim([min(f)/(1-0.1),max(f)])
		#plt.ylim([min(f)*b/(1-0.1),max(f)*b])
		#plt.errorbar(x,y,0.017,0.16*0.02,'k')

		plt.xlabel(label[1]);plt.ylabel(label[0])

		sb_red = '%.2g' % sb
		r_str = str(R); r_red = r_str[0]+r_str[1]; 
		for i in range(2,len(r_str)):
			r_red = r_red + r_str[i]
			if r_str[i] != '9':
				break
		r_sux = str(len(r_red.split('.')[1]))

		fmt_b = fix(b,sb)

		plt.title(('$b$ $=$ '+fmt_b[0]+' $±$ ' +fmt_b[1]+ '       $R$ = %s') % (r_red))
		
		if text:
			print ('Datos Regresion ponderada sen a; ', nomefigura,'\n','--------------------------------------------------------')
			print ('b = ',b,'        s(b) = ',sb)
			print ('R = ',R)
			print ('s = ',s)
			print ('SUXERENCIAS:\n'+'sb: [' + sb_red +']\n'+'R: [' + r_red +']')
			print ('--------------------------------------------------------')

		plt.grid(False)
		plt.show(False)

	if os.path.isdir('graph') == False:
		os.mkdir('graph')
	plt.savefig('graph/' + nomefigura)
	return(b,sb,R)
