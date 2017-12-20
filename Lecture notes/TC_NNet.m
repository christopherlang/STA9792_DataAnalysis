
% Input Training Vectors
X=[
0.9	0.9	0.1	0.1	0.9	0.9	0.9	0.9
0.9	0.1	0.9	0.1	0.9	0.1	0.9	0.9
0.9	0.1	0.1	0.9	0.9	0.9	0.9	0.9
0.1	0.9	0.1	0.9	0.9	0.9	0.1	0.9
0.9	0.9	0.9	0.9	0.1	0.1	0.1	0.1
0.1	0.9	0.1	0.9	0.1	0.9	0.9	0.9
0.1	0.9	0.9	0.1	0.9	0.9	0.9	0.9
0.9	0.1	0.9	0.1	0.9	0.9	0.9	0.1
0.1	0.1	0.9	0.9	0.9	0.9	0.9	0.9
];
% Output Trading Vector
Y=[
0.9	0.9	0.9	0.9	0.1	0.1	0.1	0.1
];

% Increase Training Vector
numRepeats=100;
XNet=repmat(X,1,numRepeats);
YNet=repmat(Y,1,numRepeats);


% network
%net = feedforwardnet([2 2 1]);
net = feedforwardnet([2 2]);

% initial process functions
    % no data mappings or data transformation needed
    % output data was already transformed, input data is binary
net.outputs{3}.processFcns={};
net.inputs.processFcns{1};
net.inputs.processFcns{2};
net.inputs{1}.processFcns={};
net.outputs{3}.processFcns={};
net.biases{1}.learn = 1;       %by default this value is 1 (which means it changes during training)

% 'tansig' or 'logsig' or 'purelin'
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
%net.layers{3}.transferFcn = 'logsig';

% set tolerance
net.trainParam.epochs = 10000;
net.trainparam.mu=1.0000e-12;
net.trainparam.mu_max=1.0000e+12;
net.trainparam.min_grad=1.0000e-12;


% Train the network on the given inputs and outputs
view(net);                         %View the network architecture
[net, tr] = train(net,XNet,YNet);  % Train the network on the given inputs and outputs

% Get Weights
w1=net.IW{1};
w2=net.LW{2,1};
w3=net.LW{3,2};
b1=net.b{1};
b2=net.b{2};
b3=net.b{3};



% Estimate using the NNet
estY1 = net(X);


% Estimate Mathematically
estY2=w3*logsig(w2*logsig(w1*X+b1)+b2)+b3;

% Estimate Mathematically via formula
a1=w1*X+b1;
a1Adj=1./(1+exp(-a1));
a2=w2*a1Adj+b2;
a2Adj=1./(1+exp(-a2));
estY3=w3*a2Adj+b3;

disp([Y;estY1;estY2;estY3])


% New Data
%y1= C = 0.9
%y2 = T = 0.1
x1=[0.9	0.9	0.9	0.9	0.1	0.1	0.9	0.1	0.1]';
x2=[0.1	0.9	0.9	0.1	0.9	0.1	0.1	0.9	0.1]';
y1=net(x1);
y2=net(x2);

% display results
disp([y1 y2])

