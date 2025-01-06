import numpy as np
import random



class MortgageModel:

    def __init__(self, ltv = 0, income = 0, credit_score = 0, trend = 0):
        """
            States are initialized as dictionary with  default values of zero
            States Space:
                - Loan to Value
                - Customers credit score (Range)
                - Customers income (Range)
                - Repayment trend (Categorical)

            Actions are initialized as a list of tuples
            Action Space
                - Approval decision: Approve (1) or Deny (0)
                - Interest rate level: Low (0) or High (1) if approved
        """

        # Initialise state values
        self.state = {
            'ltv': ltv,         # Loan-to-value ratio
            'income': income,   # Annual income
            'credit_score': credit_score,   # Credit score
            'trend': trend      # Repayment trend (timely=0, delayed=1, failed=2)
        }

        # Initialise Actions:
        self.actions = [(0, None), (1, 0), (1, 1)]

    def transition(self, action, exogenous_info):
        """
        Transition function simulates the state change given an action and exogenous information.
        - action: Tuple (approve, interest_rate_level)
        - exogenous_info: Tuple (house_value_change, credit_change, income_change)
        Returns the Updated state
        """
        ltv, income, credit_score, trend = self.state.values()
        approve, rate = action
        house_value_change, credit_change, income_change = exogenous_info
        
        credit = credit_score 

        if approve == 1:
            # Update LTV
            loan_balance = ltv * 100000
            loan_balance -= 5000 
            house_value = 100000 * (1 + house_value_change)
            new_ltv = loan_balance / house_value
            ltv = np.clip(new_ltv, 0.5, 1.5)  # Ensure LTV is within a realistic range

            # Update credit and income
            credit = np.clip(credit_score + credit_change, 300, 850)  # Credit score bounds
            income += income_change

            # Update repayment trend based on past trend
            if trend == 2:  # Already in default
                new_trend = 2
            elif random.random() < 0.2:  # 20% chance of delayed payment if not default
                new_trend = 1
            else:
                new_trend = 0
            trend = new_trend

        # Update state dictionary
        self.state = {'ltv': ltv, 'income': income, 'credit': credit, 'trend': trend}
        return self.state



    def reward(self, action, alpha=1, beta=10):
        """
        Reward function that combines profit and risk based on the action.
        - action: Tuple (approve, interest_rate_level)
        - alpha: Weight for profit
        - beta: Weight for risk
        return Reward value
        """
        ltv, income, credit, trend = self.state.values()
        approve, rate = action

        # Calculate profit if approved
        if approve == 1 and trend != 2:  # No profit if defaulted
            interest_rate = 0.05 if rate == 0 else 0.08  # Example rates
            profit = interest_rate * income
        else:
            profit = 0

        # Calculate risk (higher LTV and lower credit means more risk)
        default_risk = ltv * (800 - credit) / 1000  # Simplified risk model
        risk = default_risk * income

        # Total reward: profit minus the risk
        return alpha * profit - beta * risk


