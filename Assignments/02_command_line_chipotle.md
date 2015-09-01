## Class 2 Homework: Command Line Chipotle

**Command Line Tasks:**

1. Look at the head and the tail of **chipotle.tsv** in the **data** subdirectory of this repo. Think for a minute about how the data is structured. What do you think each column means? What do you think each row means? Tell me! (If you're unsure, look at more of the file contents.)

  ```Shell
  Micahs-MacBook-Pro:~ micah$ cd Do[Kocuments/D[KGeneralAssembly/Data\ Science/DAT8/data/
  order_id	quantity	item_name	choice_description	item_price
  1	1	Chips and Fresh Tomato Salsa	NULL	$2.39
  1	1	Izze	[Clementine]	$3.39
  1	1	Nantucket Nectar	[Apple]	$3.39
  1	1	Chips and Tomatillo-Green Chili Salsa	NULL	$2.39
  2	2	Chicken Bowl	[Tomatillo-Red Chili Salsa (Hot), [Black Beans, Rice, Cheese, Sour Cream]]	$16.98
  3	1	Chicken Bowl	[Fresh Tomato Salsa (Mild), [Rice, Cheese, Sour Cream, Guacamole, Lettuce]]	$10.98
  3	1	Side of Chips	NULL	$1.69
  4	1	Steak Burrito	[Tomatillo Red Chili Salsa, [Fajita Vegetables, Black Beans, Pinto Beans, Cheese, Sour Cream, Guacamole, Lettuce]]	$11.75
  4	1	Steak Soft Tacos	[Tomatillo Green Chili Salsa, [Pinto Beans, Cheese, Sour Cream, Lettuce]]	$9.25
  Micahs-MacBook-Pro:data micah$ tail chipotle.tsv
  1831	1	Carnitas Bowl	[Fresh Tomato Salsa, [Fajita Vegetables, Rice, Black Beans, Cheese, Sour Cream, Lettuce]]	$9.25
  1831	1	Chips	NULL	$2.15
  1831	1	Bottled Water	NULL	$1.50
  1832	1	Chicken Soft Tacos	[Fresh Tomato Salsa, [Rice, Cheese, Sour Cream]]	$8.75
  1832	1	Chips and Guacamole	NULL	$4.45
  1833	1	Steak Burrito	[Fresh Tomato Salsa, [Rice, Black Beans, Sour Cream, Cheese, Lettuce, Guacamole]]	$11.75
  1833	1	Steak Burrito	[Fresh Tomato Salsa, [Rice, Sour Cream, Cheese, Lettuce, Guacamole]]	$11.75
  1834	1	Chicken Salad Bowl	[Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Guacamole, Lettuce]]	$11.25
  1834	1	Chicken Salad Bowl	[Fresh Tomato Salsa, [Fajita Vegetables, Lettuce]]	$8.75
  1834	1	Chicken Salad Bowl	[Fresh Tomato Salsa, [Fajita Vegetables, Pinto Beans, Lettuce]]	$8.75
  ```

  - Each row is an order line item.
  - Each column is an attribute of the order line item.
  - Data Dictionary:

| Column Name | Definition |
| ----------- | ---------- |
| order_id | Id of the overall order to which this order  line item belongs |
| quantity | Quantity of items ordered |
| item_name | Name of the main item ordered |
| choice_description | Detailed description of the order including sides/options |
| item_price | Total price for the order line item |

1. This should be here but markdown I can't get numbered list to start at 2 and tables seem to not allow indention.

2. How many orders do there appear to be?

  ```Shell
  Micahs-MacBook-Pro:data micah$ cut -f1 chipotle.tsv | sort -g | tail -n 1
  1834
  ```

3. How many lines are in this file?

  ```Shell
  Micahs-MacBook-Pro:data micah$ wc -l chipotle.tsv | sed -e 's/^[[:space:]]*//' | cut -d ' ' -f1
  4623
  ```

4. Which burrito is more popular, steak or chicken?

  ```Shell
  Micahs-MacBook-Pro:data micah$ grep 'Chicken Burrito' chipotle.tsv |  cut -f2 | sort | uniq -c | sed -e 's/^ *//' | awk 'BEGIN {total = 0} {print "+",$1,"*",$2; total+=$
  $1*$2} END{print "total is ", total}'
  + 521 * 1
  + 28 * 2
  + 2 * 3
  + 2 * 4
  total is  591

  Micahs-MacBook-Pro:data micah$ grep 'Steak Burrito' chipotle.tsv |  cut -f2 | sort | uniq -c | sed -e 's/^ *//' | awk 'BEGIN {total = 0} {print "+",$1,"*",$2; total+=$1*
  *$2} END{print "total is ", total}'
  + 352 * 1
  + 14 * 2
  + 2 * 3
  total is  386
  ```

  _Winner winner, chicken dinner._

5. Do chicken burritos more often have black beans or pinto beans?

  ```Shell
  Micahs-MacBook-Pro:data micah$ grep -i 'Chicken Burrito' chipotle.tsv | grep -i 'Black Beans' | grep -iv 'Pinto Beans' |  cut -f2 | sort | uniq -c | awk 'BEGIN {total = 0} {print "+",$1,"*",$2; total+=$1*$2} END{print "total is ", total}'
  + 241 * 1
  + 19 * 2
  + 2 * 4
  total is  287
  Micahs-MacBook-Pro:data micah$ grep -i 'Chicken Burrito' chipotle.tsv | grep -i 'Pinto Beans' | grep -iv 'Black Beans' |  cut -f2 | sort | uniq -c | awk 'BEGIN {total = 0} {print "+",$1,"*",$2; total+=$1*$2} END{print "total is ", total}'
  + 82 * 1
  + 3 * 2
  total is  88
  Micahs-MacBook-Pro:data micah$ grep -i 'Chicken Burrito' chipotle.tsv | grep -i 'Black Beans' | grep -i 'Pinto Beans' |  cut -f2 | sort | uniq -c | awk 'BEGIN {total = 0} {print "+",$1,"*",$2; total+=$1*$2} END{print "total is ", total}'
  + 20 * 1
  total is  20
  ```
  _Black beans are almost 3x more popular and ~5% get both_

6. Make a list of all of the CSV or TSV files in the DAT8 repo (using a single command). Think about how wildcard characters can help you with this task.

  ```Shell
  find . -name '*.?sv'
  ./data/airlines.csv
  ./data/chipotle.tsv
  ./data/sms.tsv
  ```

7. Count the approximate number of occurrences of the word "dictionary" (regardless of case) across all files in the DAT8 repo.

  ```Shell
  Micahs-MacBook-Pro:DAT8 micah$ pwd
  /Users/micah/Documents/GeneralAssembly/Data Science/DAT8
  Micahs-MacBook-Pro:DAT8 micah$ grep -iRho 'dictionary' . | wc -l
        13
  ```

8. **Optional:** Use the the command line to discover something "interesting" about the Chipotle data. Try using the commands from the "advanced" section!

  We'll go get the full menu.
  ```Shell
  Micahs-MacBook-Pro:data micah$ cut -f2,3,5 chipotle.tsv | tail -n +2 | sed -e 's/\$//' | awk -F"\t" '{ print $2, " $", $3/$1 }' | sort | uniq
  6 Pack Soft Drink  $ 6.49
  Barbacoa Bowl  $ 11.48
  Barbacoa Bowl  $ 11.49
  Barbacoa Bowl  $ 11.75
  Barbacoa Bowl  $ 8.69
  Barbacoa Bowl  $ 8.99
  Barbacoa Bowl  $ 9.25
  Barbacoa Burrito  $ 11.08
  Barbacoa Burrito  $ 11.48
  Barbacoa Burrito  $ 11.75
  Barbacoa Burrito  $ 8.69
  Barbacoa Burrito  $ 8.99
  Barbacoa Burrito  $ 9.25
  Barbacoa Crispy Tacos  $ 11.48
  Barbacoa Crispy Tacos  $ 11.75
  Barbacoa Crispy Tacos  $ 8.99
  Barbacoa Crispy Tacos  $ 9.25
  Barbacoa Salad Bowl  $ 11.89
  Barbacoa Salad Bowl  $ 9.39
  Barbacoa Soft Tacos  $ 11.48
  Barbacoa Soft Tacos  $ 11.75
  Barbacoa Soft Tacos  $ 8.99
  Barbacoa Soft Tacos  $ 9.25
  Bottled Water  $ 1.09
  Bottled Water  $ 1.5
  Bowl  $ 7.4
  Burrito  $ 7.4
  Canned Soda  $ 1.09
  Canned Soft Drink  $ 1.25
  Carnitas Bowl  $ 11.08
  Carnitas Bowl  $ 11.48
  Carnitas Bowl  $ 11.75
  Carnitas Bowl  $ 8.99
  Carnitas Bowl  $ 9.25
  Carnitas Burrito  $ 11.08
  Carnitas Burrito  $ 11.48
  Carnitas Burrito  $ 11.75
  Carnitas Burrito  $ 8.69
  Carnitas Burrito  $ 8.99
  Carnitas Burrito  $ 9.25
  Carnitas Crispy Tacos  $ 11.75
  Carnitas Crispy Tacos  $ 8.99
  Carnitas Crispy Tacos  $ 9.25
  Carnitas Salad  $ 8.99
  Carnitas Salad Bowl  $ 11.89
  Carnitas Salad Bowl  $ 9.39
  Carnitas Soft Tacos  $ 11.75
  Carnitas Soft Tacos  $ 8.99
  Carnitas Soft Tacos  $ 9.25
  Chicken Bowl  $ 10.58
  Chicken Bowl  $ 10.98
  Chicken Bowl  $ 11.25
  Chicken Bowl  $ 8.19
  Chicken Bowl  $ 8.49
  Chicken Bowl  $ 8.5
  Chicken Bowl  $ 8.75
  Chicken Burrito  $ 10.58
  Chicken Burrito  $ 10.98
  Chicken Burrito  $ 11.25
  Chicken Burrito  $ 8.19
  Chicken Burrito  $ 8.49
  Chicken Burrito  $ 8.75
  Chicken Crispy Tacos  $ 10.98
  Chicken Crispy Tacos  $ 11.25
  Chicken Crispy Tacos  $ 8.49
  Chicken Crispy Tacos  $ 8.75
  Chicken Salad  $ 10.98
  Chicken Salad  $ 8.19
  Chicken Salad  $ 8.49
  Chicken Salad Bowl  $ 11.25
  Chicken Salad Bowl  $ 8.75
  Chicken Soft Tacos  $ 10.98
  Chicken Soft Tacos  $ 11.25
  Chicken Soft Tacos  $ 8.49
  Chicken Soft Tacos  $ 8.75
  Chips  $ 1.99
  Chips  $ 2.15
  Chips and Fresh Tomato Salsa  $ 2.29
  Chips and Fresh Tomato Salsa  $ 2.39
  Chips and Fresh Tomato Salsa  $ 2.95
  Chips and Guacamole  $ 3.89
  Chips and Guacamole  $ 3.99
  Chips and Guacamole  $ 4
  Chips and Guacamole  $ 4.25
  Chips and Guacamole  $ 4.45
  Chips and Mild Fresh Tomato Salsa  $ 3
  Chips and Roasted Chili Corn Salsa  $ 2.95
  Chips and Roasted Chili-Corn Salsa  $ 2.39
  Chips and Tomatillo Green Chili Salsa  $ 2.95
  Chips and Tomatillo Red Chili Salsa  $ 2.95
  Chips and Tomatillo-Green Chili Salsa  $ 2.39
  Chips and Tomatillo-Red Chili Salsa  $ 2.39
  Crispy Tacos  $ 7.4
  Izze  $ 3.39
  Nantucket Nectar  $ 3.39
  Salad  $ 7.4
  Side of Chips  $ 1.69
  Steak Bowl  $ 11.08
  Steak Bowl  $ 11.48
  Steak Bowl  $ 11.75
  Steak Bowl  $ 8.69
  Steak Bowl  $ 8.99
  Steak Bowl  $ 9.25
  Steak Burrito  $ 11.08
  Steak Burrito  $ 11.48
  Steak Burrito  $ 11.75
  Steak Burrito  $ 8.69
  Steak Burrito  $ 8.99
  Steak Burrito  $ 9.25
  Steak Crispy Tacos  $ 11.75
  Steak Crispy Tacos  $ 8.69
  Steak Crispy Tacos  $ 8.99
  Steak Crispy Tacos  $ 9.25
  Steak Salad  $ 8.69
  Steak Salad  $ 8.99
  Steak Salad Bowl  $ 11.89
  Steak Salad Bowl  $ 9.39
  Steak Soft Tacos  $ 11.48
  Steak Soft Tacos  $ 11.75
  Steak Soft Tacos  $ 8.99
  Steak Soft Tacos  $ 9.25
  Veggie Bowl  $ 10.98
  Veggie Bowl  $ 11.25
  Veggie Bowl  $ 8.49
  Veggie Bowl  $ 8.75
  Veggie Burrito  $ 10.98
  Veggie Burrito  $ 11.25
  Veggie Burrito  $ 8.49
  Veggie Burrito  $ 8.75
  Veggie Crispy Tacos  $ 8.49
  Veggie Salad  $ 8.49
  Veggie Salad Bowl  $ 11.25
  Veggie Salad Bowl  $ 8.75
  Veggie Soft Tacos  $ 11.25
  Veggie Soft Tacos  $ 8.49
  Veggie Soft Tacos  $ 8.75
  ```
  It's a bit unclear where the data comes from but either prices are variable
  over time or over location. The ~$2.50 price difference between versions
  of something (e.g. on Veggie Soft Tacos at $8.75 and one at $11.25 is a function
  of whether there is Guacamole or not; verified by grepping for this specifically).
