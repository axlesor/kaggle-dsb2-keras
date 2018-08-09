# Start with 30 iterations
# run with optirun if bumblebee is used in Linux (Optimus laptop)
[optirun] python mytrain.py
# Cross validate next 15 iterations with different divisions between train & validation
[optirun] python mytrain_pretrain.py
[optirun] python mytrain_pretrain.py
[optirun] python mytrain_pretrain.py
[optirun] python mytrain_pretrain.py
[optirun] python mytrain_pretrain.py
# Generally 120th iteration is good for a result
[optirun] python mytrain_pretrain.py
