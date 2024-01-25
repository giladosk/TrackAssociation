import numpy as np

# cost_matrix_minimise = np.array([[108, 125, 150],
#                                  [150, 135, 175],
#                                  [122, 148, 250]])  # we know the min assignments are: (0, 2), (1, 1), (2, 0)
#
# cost_matrix_minimise = np.array([[9, 2, 7, 8],
#                                  [6, 4, 3, 7],
#                                  [5, 8, 1, 8],
#                                  [7, 6, 9, 4]])  # we know the min assignments are:...
# cost_matrix = np.amax(cost_matrix_minimise) - cost_matrix_minimise
# cost_matrix = np.array([[2, 4, 3, 5, 4],
#                         [7, 4, 6, 8, 4],
#                         [2, 9, 8, 10, 4],
#                         [8, 6, 12, 7, 4],
#                         [2, 8, 5, 8, 8]])
# cost_matrix = np.array([[4, 3, 1, 9],
#                         [6, 1, 1, 10]])  # less tracks than new items in new frame
cost_matrix = np.array([[4, 3],
                        [6, 1],
                        [0, 9]])
print(f'\ncost_matrix: \n{cost_matrix}\n')

# rows = bidders (or owners), columns = goods
num_bidders = cost_matrix.shape[0]
num_goods = cost_matrix.shape[1]

owners_allocated_to_goods = dict.fromkeys(range(num_goods))
association_matrix = np.zeros((num_bidders, num_goods), dtype=int)
best_prices = [0] * num_goods
bidders_queue = list(range(num_bidders))
# epsilon_price = round(np.log10(np.mean(cost_matrix)))
# epsilon_price = 1 / num_bidders + 1
epsilon_price = 0.1

print('start bidding...')
while len(bidders_queue) > 0:
    bidder = bidders_queue.pop(0)  # take the first bidder in line
    print(f'{bidder=}')
    desired_good = np.argmax(benefits := (cost_matrix[bidder, :] - best_prices))
    price_rise = benefits[desired_good]
    if not association_matrix[:, desired_good].any():
        association_matrix[bidder, desired_good] = 1  # first time assignment of a good
        best_prices[desired_good] += epsilon_price
    if price_rise > epsilon_price and association_matrix[bidder, desired_good] == 0:  # re-assignment for higher bid
        # put previous bidder in end of queue, and set new bidder as the owner
        previous_owner = association_matrix[:, desired_good].argmax()
        bidders_queue.append(previous_owner)
        association_matrix[previous_owner, desired_good] = 0
        association_matrix[bidder, desired_good] = 1
        best_prices[desired_good] += epsilon_price
    print(association_matrix)

total_price = 0
print('\nfinal results:')
print(f'association_matrix=\n{association_matrix}')
for item in range(num_goods):
    if association_matrix[:, item].sum() > 0:
        owner = association_matrix[:, item].argmax()
        total_price += cost_matrix[owner, item]
        print(f'{item=}: {owner=}')
    else:
        print(f'{item=}: no owner')
for owner in range(num_bidders):
    if association_matrix[owner, :].sum() == 0:
        print(f'no item for {owner=}')

print(f'{total_price=}')

    # if owners_allocated_to_goods[desired_good] is None:
    #     owners_allocated_to_goods[desired_good] = bidder  # first time assignment of a good
    #     best_prices[desired_good] += epsilon_price
    # if price_rise > epsilon_price and bidder != owners_allocated_to_goods[desired_good]:  # re-assignment for higher bid
    #     # put previous bidder in end of queue, and set new bidder as the owner
    #     bidders_queue.append(owners_allocated_to_goods[desired_good])
    #     owners_allocated_to_goods[desired_good] = bidder
    #     best_prices[desired_good] += epsilon_price
    # print([f'{x[0]}: {x[1]}, {y}' for x, y in zip(owners_allocated_to_goods.items(), best_prices)])

# total_price = 0
# print('\nfinal results:')
# for item, owner in owners_allocated_to_goods.items():
#     if owner is None:
#         print(f'{owner=} for {item=}')
#         continue
#     print(f'{owner=} for {item=} cost={cost_matrix[owner, item]}')
#     total_price += cost_matrix[owner, item]
# print(f'{total_price=}')

last_line = 0
