# buyLotsOfFruit.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
To run this script, type

  python buyLotsOfFruit.py

Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""
from __future__ import print_function

fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75,
               'limes': 0.75, 'strawberries': 1.00}


def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples

    Returns cost of order
    """
    totalCost = 0.0
    "*** YOUR CODE HERE ***"
    ####################################################################################################################
    # For each tuple (fruit, pound) in the list, calculate the cost of the fruit and add it to the total cost
    for fruitTuple in orderList:

        # Store the name of the fruit from the tuple (fruit, pound)
        fruitToSearch = fruitTuple[0]  # Store the data from "fruitTuple" at index 0 in "fruitToSearch"

        # Store the number of pounds of the fruit from the tuple (fruit, pound)
        fruitPounds = fruitTuple[1]  # Store the data from "fruitTuple" at index 1 in "fruitPounds"

        # Initialize the cost of the fruit per pound to $0.00
        fruitCostPP = 0.0

        # If the fruit is in the dictionary, calculate the cost of the fruit and add it to the total cost
        if fruitToSearch in fruitPrices.keys():  # If "fruitToSearch" is a key in the dictionary "fruitPrices"...

            # Store the cost of the fruit per pound from the dictionary
            fruitCostPP = fruitPrices[fruitToSearch]  # Store the value from "fruitPrices" with key "fruitToSearch"

            # Calculate the cost of the fruit
            fruitCost = fruitPounds * fruitCostPP

            # Add the cost of the fruit to the total cost
            totalCost = totalCost + fruitCost

        # If the fruit is NOT in the dictionary, print an error message and return nothing from the function
        else:  # If "fruitToSearch" is NOT a key in the dictionary "fruitPrices"...

            # Print an error message
            print('ERROR - Invalid Fruit')

            # Leave the function and return nothing
            return None
    ####################################################################################################################
    return totalCost


# Main Method
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)]
    print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))
