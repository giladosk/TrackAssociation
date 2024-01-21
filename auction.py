import numpy as np

cost_matrix_minimise = np.array([[108, 125, 150],
                                 [150, 135, 175],
                                 [122, 148, 250]])  # we know the min assignments are: (0, 2), (1, 1), (2, 0)

cost_matrix = np.amax(cost_matrix_minimise) - cost_matrix_minimise
print(f'{cost_matrix}')

# rows = bidders (or owners), columns = goods
num_bidders = cost_matrix.shape[0]
num_goods = cost_matrix.shape[1]

owners_allocated_to_goods = {key: {'owner': key, 'best_price': 0} for key in range(num_goods)}
bidders_queue = list(range(num_bidders))
# delta_price = 1 / num_bidders + 1
delta_price = 0

while len(bidders_queue) > 0:
    bidder = bidders_queue.pop(0)  # take the first bidder in line
    print(f'{bidder=}')
    desired_good = np.argmax(cost_matrix[bidder, :])
    price_rise = cost_matrix[bidder, desired_good] - owners_allocated_to_goods[desired_good]['best_price']
    if price_rise > 0:
        # put previous bidder in end of queue, and set new bidder as the owner
        bidders_queue.append(owners_allocated_to_goods[desired_good]['owner'])
        owners_allocated_to_goods[desired_good]['owner'] = bidder
        owners_allocated_to_goods[desired_good]['best_price'] += price_rise + delta_price
    print(owners_allocated_to_goods)

last_line = 0
