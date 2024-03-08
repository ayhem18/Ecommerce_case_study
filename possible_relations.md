# Data
1. products: contains the product identifier, the dimensions, the weight, category, name and description, the number of pictures
2. customers: id,

# Relations:

## Products: 
* products with more pictures are more popular (consider aggregating the product categories) current no 
* products with longer (supposedly more engaging) description are more popular (maybe consider the aggregation of product categories) (investigating)
* weight might be somehow related to shipping: heavier weights take longer to ship (grouped together maybe)
* we have enough statistical evidence to accept that the number of photos does not affect a product popularity

## Customer
* What about the mean and variance of spending / number of products purchased by customers in the 2 major cities (Sao Paulo and Rio De janeiro) ? can any meaningful insights be derived from that ?
* can we find different means for different regions in general: states / or simply groups created from the geo coordinates? (might require some geospatial processing)
* only 2k customers order multiple times
* the difference in the average spending does not seem as significant across the 3 major states

## Payment
* Are customers inclined to spend more with more payment installements ?
## Sellers
* how can we estimate profits ? : either by calculating the cost of the order - the value of the payment: these are equivalent
* the price is determined by the seller
* most products are sold by only one seller !!!
* only $\approx 1.2$ k products are sold by multiple sellers (at least 2)
* to understand the effect of geo location on sellers' profits, we can proceed as follows: 
    1. for each product sold by more than 1 seller, say $p_i$, we consider its sellers
    2. group them by state, and calculate the average price for each state
    3. now we have created a sequence $P\_{state} = x_1, x_2, ... x_n$ where $x_i$ is the average price of $p_i$ sold by sellers belong to the state in question 

* freight value in the same-state orders vs cross-state orders !!
