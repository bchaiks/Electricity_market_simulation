'''
Compuational Simulation of Electricity market incorporating 
a regulation that was proposed in Central Ohio (and elsewhere). 

In particular, proposal was to subsidize coal-fired plants to keep 
them viable during a period where coal was much more expensive than 
natural gas, and coal-fired plants were in danger of shutting down. 
 
Idea is that it is better for these to stay open in the long run, 
and that eventually their costs will come down, and the subsidy payment 
will become a credit that is returned to consumers. 
 
In Ohio, a utility may own multiple generators.  In particular, a subsidized 
firm and a non-subsidized firm may be owned by the same holding company.  
I proved theoretically that in this case, the holding company has an incentive 
to reduce the subsidized firm's output to 0, regardless of its costs. 
 
The results of the simulation below show that for a market with 
relatively realistic characteristics, this does indeed occur. 


This code computes Nash-Cournot-Equilibrium market solutions (i.e. market 
where each supplier chooses its quantity and takes price as given by the 
market -- i.e. by the consumers' demand function)
'''
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

#### elasticity of demand * -1
g = 2.5

#### Marginal cost of subsidized firm
cs = 33.0

#### marginal cost for firm associated with subsidized firm
c_s1 = 27

#### marginal costs of other firms
c = [29.0, 25.0 , 26.0 , 27.0]

#### fixed cost for subsidized firm
fs = 1800.0
#### fixed cost affiliate firm
f_s1 = 1500.0

#### Fixed profit guaranteed to subsidized firm by the subsidy
#### covers fixed cost and gives 15% return on costs
L = 10000.0 + fs 

#### fixed cost for other firms
f = [1200.0, 1400.0, 1500.0, 1200.0]

#### Initial values for production quantities
qs = 100.0     
q_s1 = 1000.0
q = [1000.0,1000.0,1000.0,1000.0]

#### Starting total quantity so all the functions have inputs
Q = sum(q)   

########################################################################
''' 
Parameter computation for price function matching theoretical work
and definitions of these functions with and without the subsidy
Original price function is P(Q) = t[0] - t[1]*Q**g
solve for t[0] and t[1] using the reference values: 
P(9000) = 40 and p(20000) = 20

Solve the model with the price augmented to incorporate the 
effects of the subsidy payment on consumers' willingness to pay 
to consume any given quantity: 

P_alt(Q) = Q/(Q - qs) * (t[0] - t[1]*Q**g) - (L + cs* qs)/Q

The intuition for this is that amount that PRODUCERS expect to be 
able to sell at a given price is shifted lower.
''' 
refd1 = 9000
refd2 = 20000
refp1 = 40
refp2 = 20

A = [[1, -refd1**g],[1, -refd2**g]]
G = np.linalg.inv(A)
t = np.dot(G, [refp1, refp2])


def Price_s(x):
    return (Q+ x[0]+ x[1])/(Q+ x[0])*(t[0] - t[1]*((Q+x[0]+x[1])**2.5)) - 
    		(L + cs*x[1])/(Q + x[0])   

def Price(q_i):
   return (BQ+q_i)/(BQ+q_i - qs)*(t[0] - t[1]*((BQ+q_i)**2.5)) - 
   			(L + cs*qs)/(BQ+q_i - qs)

def Profit_subsd(x):    
    return -1*((Price_s(x) - c_s1)*x[0] + L - fs - f_s1)

def Profit(q_i):
    return -1*((Price(q_i) - c_i)*q_i - f_i)

########################################################################


#### 2500 is a rough estimate of subsidized firm production before 
#### it needed to be subsidized 
ref_quant = 2500

#### consumer surplus if the subsidized firm exits the market 
#### (without the subsidy) -- this is from an auxiliary computation
Un_Surplus = 115158.8


#### initialize places to store outputs
Subsd_Profits = [[],[],[],[],[]]
Subsd_Surps = []

########################################################################
'''
Begin main loops:
	
	Sequentially maximizes each firm's profit (by altering production quantity), 
	taking the other firms' production as a given parameter.  Process continues 
	until no change (greater than tol) in any firm's quantity.   
	
	The resulting set of production quantities is an approximate Nash-Cournot 
	equilibrium for the market with the firms, price, and profit defined above.
	
	Solves the nonlinear optimization using sequential linear approximation
	(COBYLA).  This is a small instance, so it works, could replace with 
	IPOPT for higher quality solutions for larger instances with more 
	complicated price functions.  
	
	The theoretical analysis indicates that as the subsidized firm increases 
	its production, all other firms' profits decrease.  The outer loop thus 
	increases the lower bound on this output in each stage to see how the 
	equilibrium profits of the other firms' are affected.    

'''
tol = 0.01
for w in range(11): 
	big_change_q = 2
	old_val= 0.0
	old_vals = [0.0,0.0,0.0,0.0]    

	### lower bound on subsidized production 
	### as percentage of reference
	MINQ = w/10. * ref_quant

	cons_subsd = ({'type': 'ineq', 'fun': lambda x: x[0]},
	# add MINQ to constrain subsidized guy to produce at least K mwh...
			{'type': 'ineq', 'fun': lambda x: x[1] - MINQ }, 
			{'type': 'ineq', 'fun': lambda x: 4000.0-x[0]},
			{'type': 'ineq', 'fun': lambda x: 4000.0-x[1]})

	cons = ({'type': 'ineq', 'fun': lambda x:  x},
			{'type': 'ineq', 'fun': lambda x: 4000.0 - x})
	
	##### Reset the starting values each time through the for-loop #####
	qs = 100.0     
	q_s1 = 1000.0
	q = [1000.0,1000.0,1000.0,1000.0]
	
	profits = [0,0,0,0,0]
	## uncomment if want to keep track of times through the while loop
	#k = 1
	oldqs= 0.0
	oldqs1 = 0.0
	oldq=[0.0,0.0,0.0,0.0]

	while big_change_q > tol:
	
		res_sub = opt.minimize(Profit_subsd, (q_s1,qs),method='COBYLA', 
												constraints = cons_subsd)
		q_s1 = res_sub.x[0]
		qs = res_sub.x[1]
		val = -1*res_sub.fun
		changeq_s1 = np.abs(q_s1 - oldqs1)
		big_change_q = changeq_s1
		old_val = val  
		oldqs1 = q_s1
		profits[4] = val
		

		for i in range(0,4):
			BQ = q_s1 +qs
			#### Computes other firms' current output to use 
			#### as input to firm i's profit max in this round 
			for j in range (0,4):
				if j == i:
					continue
				BQ = BQ+q[j]
			
			#### sets the costs that appear in the 
			#### price functions above
			c_i = c[i]
			f_i = f[i]
			  
			res = opt.minimize(Profit,q[i], method ='COBYLA', constraints = cons)
			q[i] = res.x
			val = -1*res.fun
			changeq = np.abs(q[i] - oldq[i])
			old_vals[i] = val
			oldq[i] = q[i]
			if changeq > big_change_q:
				big_change_q = changeq
			profits[i] = val
		Q = sum(q) 
		#k = k+1

	#### Total final quantity produced in the market
	QQ = Q+q_s1+qs
	
	#### Computes consumer surplus -- integral of the  
	#### price function up to equilibrium quantity, minus total  
	#### payment and minus the extra subsidy payment.  
	surplus = (t[0]*QQ - (t[1]/3.5)*(QQ**3.5)) - (t[0] - t[1]*(QQ**2.5) +  
				(L - (t[0] - t[1]*(QQ**2.5)-cs)*qs)/QQ) * QQ
	
	# keeping track of firm profits
	for i in range(5):
		Subsd_Profits[i].append(profits[i])
	
	# keeping track of consumer surplus
	Subsd_Surps.append(surplus)

#### Print the subsidized firm's production  
#### (expect it to always be at the lower bound)
print('qs:', qs)	

#### Print firm profits
for i in range(4):
	print(f'\n Firm {i+1} Subsidized Profits:', np.around(Subsd_Profits[i],2))
print('\n Holding Company Subsidized Profits:', np.around(Subsd_Profits[4],2))

#### Print consumer surpluses
print('\n Subsidized Consumer Surplus:', np.around(Subsd_Surps,2))
print('\n Unsubsidized Consumer Surplus (High Price):',  Un_Surplus)	

##### make a vector of the unsubsidized consumer surplus to plot
Un = np.zeros(11)
for i in range(11):
	Un[i] = Un_Surplus
	
#### Indices represent %-age of the reference production value 
#### that the subsidized firm produces
indices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0]

########################################################################
######## Plot Outcomes
########################################################################

fig, axs = plt.subplots(2) 

fig.suptitle(f'Subsidized Firm Marginal Cost = 33')
axs[0].plot(indices, Subsd_Profits[0], '+')
axs[0].plot(indices, Subsd_Profits[1],'o')
axs[0].plot(indices, Subsd_Profits[2], 'x')
axs[0].plot(indices, Subsd_Profits[3],'d')
axs[0].plot(indices, Subsd_Profits[4], '4')

axs[0].set(ylabel='Firm Profits')


axs[0].legend(('Firm 1','Firm 2','Firm 3','Firm 4','Holding Company'), 
				loc='upper right', fontsize = 'x-small')
	
axs[1].plot(indices, Un, '-.')	
axs[1].plot(indices, Subsd_Surps)
axs[1].set(xlabel='Subsidized Firm Production (% of pre-subsidy value)', 
				ylabel = 'Consumer Surplus')
axs[1].legend(('Without Subsidy','With Subsidy'), loc = 'center right', 
				fontsize = 'x-small')
plt.show()


