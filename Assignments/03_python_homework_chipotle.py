'''
Python Homework with Chipotle data; answers by Micah
data comes from https://github.com/TheUpshot/chipotle
'''
import requests
import csv
from collections import defaultdict
import pprint

#Instantiating a pretty printer for future use
pp = pprint.PrettyPrinter(indent=4)


'''
BASIC LEVEL
PART 1: Read in the file with csv.reader() and store it in an object called 'file_nested_list'.
Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.
      https://docs.python.org/2/library/csv.html

Two functions created; one for pulling data from the web and one from the file
downloaded to local file systme (in current directory).

Each function "opens file/stream" then creates a csv reader with tab delimeter
Then reads each row in in the csv reader into a list.
This strikes me as a bad idea if file is indeterminably large but it's not
so I'm moving on for now.
NOTE; the tricky part of pulling the data from web is that I had to splitlines
before reading it in via csv.reader
'''

def get_chipotle_data_web():
    r = requests.get("https://raw.githubusercontent.com/TheUpshot/chipotle/master/orders.tsv")
    file_data = csv.reader(r.text.splitlines(), delimiter='\t')
    file_nested_list = [row for row in file_data]
    return(file_nested_list)

def get_chipotle_data_local():
    with open('chipotle.tsv', 'rU') as r:
        file_data = csv.reader(r, delimiter='\t')
        file_nested_list = [row for row in file_data]
        return(file_nested_list)

# Using web version so I don't have to worry about being in the correct directory

file_nested_list = get_chipotle_data_web()

'''
BASIC LEVEL
PART 2: Separate 'file_nested_list' into the 'header' and the 'data'.

The header is first row
The data is everything on second row and behond.
'''
header = file_nested_list[0]
data = file_nested_list[1:]


'''
INTERMEDIATE LEVEL
PART 3: Calculate the average price of an order.
Hint: Examine the data to see if the 'quantity' column is relevant to this calculation.
Hint: Think carefully about the simplest way to do this!

Answer
Splitting this out for obviousness so excuse verbosesness.
1) Get a list of all the item prices (which already accounts for quantity) as a
 number; the trickiness here is that we have to remove the first char ($) to be
 able to convert to float
2) Get a list of all the order ids as ints
3) Get a list of distinct order ids by converting list to set
4) Average order size is sum or item_price / number of orders (aka # unique
 order ids)
'''

order_line_item_prices = [float(row[4][1:]) for row in data]
order_line_item_order_ids = [int(row[0]) for row in data]
unique_order_ids = set(order_line_item_order_ids)
average_order_price = sum(order_line_item_prices) / len(unique_order_ids)
print("Average Price of an Order: $", average_order_price)

'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.

Answer:
Did this two ways
1) For loop: create a set for unique soda names; loop through rows; see if
   item name has either Soda or Soft Drink in it and if so add the choice
   description (without first&last characters which are brackets)
2) as a set comprehension that basically does the same thing
'''
sodas = set()
for row in data:
    item_name = row[2]
    if 'Soda' in item_name or 'Soft Drink' in item_name:
        soda_name = row[3][1:-1]
        sodas.add(soda_name)

sodas_comprehension = {row[3][1:-1] for row in data 
    if 'Soda' in row[2] or 'Soft Drink' in row[2]}

print()
print("Unique Sodas:")
pp.pprint(sodas_comprehension)

'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!

Answer:
To compute the number of toppings, I decided easiest was to count the # commas
    and add one (commas being the separator for toppings)
So the algorithm is just to get a list that has the # toppings for each row
    that has 'Burrito' in the item_name and then average equals sum of the
    list divided by # items in list (len)
'''

burrito_toppings_counts = [row[3].count(',')+1 
    for row in data 
    if 'Burrito' in row[2]]
burrito_toppings_average = sum(burrito_toppings_counts) / len(burrito_toppings_counts)
print()
print("Average # toppings / burrito = ", burrito_toppings_average)

'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.

Answer: use defaultdict to initialize each new item in the dictionary with
    function int (which results in int with value of 0) and then have a for
    loop that looks at each row in data and checks to see if 'Chip' is in 
    item_name and if so set the value of the value for key of item_description
    to the current value (initialized to 0 by defaultdict if this is a new value)
    plus the current row's quantity (convered to int)
'''

chip_order_totals = defaultdict(int)
for row in data:
    if 'Chip' in row[2]:
        chip_order_totals[row[2]] += int(row[1])
print()
print("Chip Order Totals by Type of Chip")
pp.pprint(chip_order_totals)

'''
BONUS: Think of a question about this data that interests you, and then answer it!

Getting the full menu as I did for the command line first via a for loop
then via a set comprehension.
'''
full_menu = set()
for row in data:
    cur_price = float(row[4][1:])/int(row[1])
    cur_menu_item = '{0}\t${1:.2f}'.format(row[2], cur_price)
    full_menu.add(cur_menu_item)

full_menu = {"{0}    ${1:.2f}".format(row[2],float(row[4][1:])/int(row[1])) for row in data}

pp.pprint(full_menu)