[TODO] #1 Complete the convolution integral for design , validate the test case.
[TODO] #2 Implement a search algorithm given probModelAX to determine the time to completion (timeModelAX) with 90% probability of confidence. 
"""
"""
class DataInput:
    def __init__(self):
        self.probModel = float(input("Enter the probability model: "))
        self.timesObs1A = list(map(float, input("Enter the times for Obstacle 1 Design A (comma-separated): ").split(',')))
        self.mean1A = sum(self.timesObs1A)/len(self.timesObs1A)
        self.timesObs2A = list(map(float, input("Enter the times for Obstacle 2 Design A (comma-separated): ").split(',')))
        self.mean2A = sum(self.timesObs2A)/len(self.timesObs1A)
        self.T = 50
        self.ti = [t for t in range(self.T)]
        
class Solution:
    def __init__(self, data):
        self.data = data
        self.f1A = [self.uniformPDF(fti, min(self.data.timesObs1A), max(self.data.timesObs1A)) for fti in self.data.ti]
        self.f2A = [self.exponentialPDF(fti, 1/self.data.mean2A) for fti in self.data.ti]
        self.convolutionA = self.compute_convolution()
        self.mean1A = self.data.mean1A
        self.mean2A = self.data.mean2A
        self.timeModelA = self.search_ppf(self.compute_cdf(), self.data.probModel, epsilon=1e-4)
        
    def exp(self, x, terms=100):

        #Compute the exponential function
        """"
     Args:
            x (float): The exponent to calculate e^x for.
            terms (int, optional): The number of terms in the series to use for the approximation. Defaults to 100.

        Returns:
            float: The value of e^x
        """    
        result = 0
        x_power = 1
        factorial = 1
        for i in range(terms):
            result += x_power / factorial
            x_power *= x
            factorial *= i + 1
        return result

    def exponentialPDF(self, t, r):
        """Compute the probability density function for the exponential probability distribution.
        Args:
            t (float): The time.
            r (float): The rate of events per unit time.
        Returns:
            float: The probability density at time t for a given rate r.
        """
        return r*self.exp(-r*t)

    def uniformPDF(self, t,a,b):
        """Compute the probability density function for the uniform probability distribution.
        Args:
            a (float): Lower bound
            b (float): Upper bound
        Returns:
            float: The probability density at time t for a given bounds [a,b].
        """
        if a <= t and t <= b:
            return (1/(b-a))
        else:
            return 0 

    def integrateTrapz(self, f,x):
        """Perform trapezoidal integration, approximating the area under the curve over width h.

        Args:
            f (Callable): A function to integrate over x.
            x (List[float]): The x-values to compute the integral over.

        Returns:
            float: The definite integral of f(x) from x[0] to x[-1], computed using the trapezoidal rule.
        """
        n = len(x)
        if n == 1:
            return 0
        h = (x[-1] - x[0])/(n-1)
        return (h/2) * (f[0] + 2 *sum(f[1:n-1]) + f[n-1])

    def compute_convolution(self):
        convolutionA = []
        """ 
        [TODO] (1a) Do for every time step t_i:
        
        """
        for t_i in range(len(convolutionA)):
         #   [TODO] (1b) Define f
            f = (self.f1A[:i+1])
           # [TODO] (1c) Define g

            g = self.f2A[i::-1]
            Cj=[]
            for j in range(len(f)):
                Cj.append(f[j]*g[j])
            convolutionA.append(self.integrateTrapz(Cj, self.data.ti[:j]))
        return convolutionA

        """
            [TODO] (1d) Form the integrand f(tau)*g(t-tau) 
        """
        I = Solution.integrateTrapz(self.exp(t_i))
        """ 
            [TODO] (1e) Integrate I over t_i[:i+1]
        """
        # ...
        return convolutionA

    def compute_cdf(self):
        A = self.integrateTrapz(self.convolutionA, self.data.ti)
        self.convolutionA = [self.convolutionA[i]/A for i in range(len(self.convolutionA))]
        return [self.integrateTrapz(self.convolutionA[:i], self.data.ti[:i]) for i in self.data.ti[1:]]

        
    def search_ppf(cdf_values, target, epsilon=1e-6):
        """
        Calculate the PPF (point percent function = inverse cuumulative distribution function [CDF])
        of a probability distribution using search.
        
        This will find the X axis value of a given y axis value input
        
        cdf_values (list): A sorted list representing the CDF from 0 to 1.
        target (float): The target probability for which the PPF is computed.
        epsilon (float): The tolerance level for the search.

        return (float): The PPF of the probability distribution.
        """
        
        """
        [TODO] (2) Implement search function here

        """
        lo = 0
        hi = len(cdf_values) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cdf_values[mid] < target:
                lo = mid
            else:
                hi = mid
        if abs(cdf_values[hi] - target) < epsilon:
            return hi
        else:
            return lo



def main():
    data_input = DataInput()
    data_analysis = Solution(data_input)
    print(f"mean1A = {round(data_analysis.mean1A,2)}")
    print(f"mean2A = {round(data_analysis.mean2A,2)}")
    print(f"mean1A + mean2A = {round(data_analysis.mean1A + data_analysis.mean2A,2)}")
    print("The probability that two events will take less than t' < t:")
    print("PrA(t'< {:2.1f} s) = {:2.2f}".format(data_analysis.timeModelA, data_input.probModel))

if __name__ == "__main__":
    main()