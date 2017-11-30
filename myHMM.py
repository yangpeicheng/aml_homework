import numpy as np
import datetime
import tushare as ts
import sys

if sys.version_info.major == 3:
    xrange = range


class myHMM(object):
    def __init__(self, tolerance = 1e-6, max_iterations=10000):
        self.tolerance=tolerance
        self.max_iter = max_iterations

    def HMMfwd(self, a, b, o, pi):
        N = np.shape(b)[0]
        T = np.shape(o)[0]
        
        alpha = np.zeros((N,T))
        alpha[:,0] = pi*b[:,o[0]]
 
        for t in xrange(1,T):
            for i in xrange(N):
                """
                TODO: Do something to update alpha[i,t]
                """
                alpha[i,t]=b[i,o[t]]*np.dot(alpha[:,t-1],a[:,i])
        return alpha 

    def HMMbwd(self, a, b, o):
        # Implements HMM Backward algorithm
        T = np.shape(o)[0]
        N = np.shape(b)[0]
        beta = np.zeros((N,T))
        c = np.ones((T))
        beta[:,T-1] = c[T-1]
    
        for t in xrange(T-2,-1,-1):
            for i in xrange(N):
                """
                TODO: Do something to update beta[i,t]
                """
                beta[i,t]=np.sum(beta[:,t+1]*a[i,:]*b[:,o[t+1]])
        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        
        
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        path = np.zeros(T)
        delta = np.zeros((N,T))
        phi = np.zeros((N,T))

        """
        TODO: implement the viterbi algorithm and return path
        """
        delta[:,0]=pi*b[:,o[0]]
        phi[:,0]=np.zeros(N)
        for i in range(1,T):
            for j in range(N):
                delta[j,i]=np.max(delta[:,i-1]*a[:,j]*b[j,o[i]])
                phi[j,i]=int(np.argmax(delta[:,i-1]*a[:,j]*b[j,o[i]]))
        path[T-1]=int(np.argmax(delta[:,T-1]))
        for i in range(1,T)[::-1]:
            path[i-1]=phi[int(path[i]),i]
        return path 

 
    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
          
        
        T = np.shape(o)[0]

        M = int(max(o))+1  

        digamma = np.zeros((N,N,T))

    
 
        np.random.seed(rand_seed)
        
 
        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))
            
            a = np.random.dirichlet(np.ones(N),size=N)
            
            b=np.random.dirichlet(np.ones(M),size=N)
        else:
            
            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
            pi=1.0/N*np.ones(N)-pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
            a=1.0/N*np.ones([N,N])-a_randomizer

            b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
            b = 1.0/M*np.ones([N,M])-b_randomizer

        
        error = self.tolerance+10
        itter = 0
        while ((error > self.tolerance) & (itter < self.max_iter)):   

            prev_a = a.copy()
            prev_b = b.copy()
    
             
            alpha = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o) 
    
            for t in xrange(T-1):
                for i in xrange(N):
                    for j in xrange(N):
                        digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
                digamma[:,:,t] /= np.sum(digamma[:,:,t])
    

            for i in xrange(N):
                for j in xrange(N):
                    digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
            digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
             
            for i in xrange(N):
                pi[i] = np.sum(digamma[i,:,0])
                for j in xrange(N):
                    a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
    	

                for k in xrange(M):
                    filter_vals = (o==k).nonzero()
                    b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
            error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
            itter += 1            
            
            if verbose:            
                print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
        return a, b, pi, alpha
        
def parseStockPrices(from_date, to_date, symbol):
    hist_prices = ts.get_k_data(symbol, from_date, to_date, ktype="D", autype="none")
    hist_prices_qfq = ts.get_k_data(symbol, from_date, to_date, ktype="D", autype="qfq")
    np_hist_prices = np.empty(shape=[len(hist_prices),7])   
    np_hist_prices[:, 0] = hist_prices_qfq["close"]
    np_hist_prices[:, 1] = hist_prices["close"]
    np_hist_prices[:, 2] = [datetime.datetime.strptime(dt, "%Y-%m-%d").toordinal() for dt in hist_prices.date.values]
    np_hist_prices[:, 3] = hist_prices["high"]
    np_hist_prices[:, 4] = hist_prices["low"]
    np_hist_prices[:, 5] = hist_prices["open"]
    np_hist_prices[:, 6] = hist_prices["volume"]
    return np_hist_prices       
        
def calculateDailyMoves(hist_prices, holding_period):
    assert holding_period > 0, "Holding something less than 0 makes no sense"
    return (hist_prices[:-holding_period,1]-hist_prices[holding_period:,1])


if __name__ == '__main__':

    hmm = myHMM()
    hist_prices = parseStockPrices('2015-10-11', '2017-11-11', '002415')  
    hist_moves = calculateDailyMoves(hist_prices,1)
    hist_Observation = np.array(list(map(lambda x: 1 if x>0 else (0 if x<0 else 2), hist_moves)))
    hist_Observation = hist_Observation[::-1]  
    print(hist_Observation)
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_Observation, 2, False, False)
    path = hmm.HMMViterbi(a, b, hist_Observation, pi_est)
    print(path)
 
