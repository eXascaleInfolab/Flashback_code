# Location Prediction over Sparse User Mobility Traces using RNNs: Flashback in Hidden States!

## Abstract
Location prediction is a key problem in human
mobility modeling, which predicts a user’s next location
 based on historical user mobility traces. As
a sequential prediction problem by nature, it has
been recently studied using Recurrent Neural Networks
 (RNNs). Due to the sparsity of user mobility
 traces, existing techniques strive to improve
RNNs by considering spatiotemporal contexts. The
most adopted scheme is to incorporate spatiotemporal
 factors into the recurrent hidden state passing
 process of RNNs using context-parameterized
transition matrices or gates. However, such a
scheme oversimplifies the temporal periodicity and
spatial regularity of user mobility, and thus cannot
fully benefit from rich historical spatiotemporal
contexts encoded in user mobility traces. Against
this background, we propose Flashback, a general
RNN architecture designed for modeling sparse
user mobility traces by doing flashbacks on hidden
states in RNNs. Specifically, Flashback explicitly
uses spatiotemporal contexts to search past
hidden states with high predictive power (i.e., historical
 hidden states sharing similar contexts as
the current one) for location prediction, which can
then directly benefit from rich spatiotemporal contexts.
 Our extensive evaluation compares Flashback
 against a sizable collection of state-of-the-art
techniques on two real-world LBSN datasets. Results
show that Flashback consistently and significantly
outperforms state-of-the-art RNNs involving
spatiotemporal factors by 15.9% to 27.6% in the
next location prediction task.

## Results

|                  | Gowalla |        |        |        | Foursquare |        |        |        |
|------------------|---------|--------|--------|--------|------------|--------|--------|--------|
|                  | Acc@1   | Acc@5  | Acc@10 | MRR    | Acc@1      | Acc@5  | Acc@10 | MRR    |
| Flashback (RNN)  | 0.1158  | 0.2754 | 0.3479 | 0.1925 | 0.2496     | 0.5399 | 0.6236 | 0.3805 |
| Flashback (LSTM) | 0.1024  | 0.2576 | 0.3317 | 0.1778 | 0.2398     | 0.5169 | 0.6014 | 0.3654 |
| Flashback (GRU)  | 0.0979  | 0.2526 | 0.3267 | 0.1731 | 0.2375     | 0.5154 | 0.6003 | 0.3631 |

## Usage

Download and store dataset files under ```./data/``` (instructions in ```./data/README.md```).

Run ```python train.py [--dataset NAME]```.

## Cite
If you find this code useful, consider to cite our paper [ijcai20.pdf](ijcai20.pdf):

Dingqi Yang , Benjamin Fankhauser, Paolo Rosso, and Philippe Cudre-Mauroux, Location Prediction over Sparse User Mobility Traces using RNNs: Flashback in Hidden States! In Proc. of the International Joint Conference on Artificial Intelligence (IJCAI'20). July 2020, Japan.
