import numpy as np

board = "r n . q k b . r\n. . . . p p . p\n. P . . . n p .\np . p P p . . .\n. . . . . . . .\n. . N . . N . .\nP P . . . P P P\nR . B Q . K . R"





print(board)
print(np.asarray([i.split(" ") for i in board.split("\n")]))
