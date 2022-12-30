function [Jsc_device,Jsc_CROWM,Jsc_total_unshadowed,Jsc_total_shadowed,Jsc_top,Jsc_bot,Voc,FF,Eff,V_mpp,J_mpp]  = Tandem3Diode_v2(IZO_thick, finger_no, Jsc_top, Jsc_bot)

warning off
syms J JH_bot JH_top Voc_var V_bot k

% Inputs
Vth = 0.02589;

% mpp calculation
epsilon = 1e-5;               % accuracy value
iter = 12;                       % maximum number of iterations
tau = (sqrt(5) - 1)/2;      % golden proportion coefficient, around 0.618
i = 0;                            % number of iterations

% Cell geometry
finger_width = 50e-4;
a = 1;
b = (1 - finger_width*finger_no)/(finger_no + 1);

% IZO_thick = 50; % in nm
IZO_res = 4e-4; % ohm.cm
Rsh_IZO = IZO_res/(IZO_thick*1e-7);
Rs_TCO_Grid = (1/12*a/b - (a/b).^2*(16/pi^5)*double(symsum((2*k + 1).^-5*tanh((2*k + 1)*pi/2*b/a), k, 0, inf)))*Rsh_IZO*a*b; % Series resistance due to the top TCO and grid
Rs_other = 1; % Series resistance due to other losses
Rs = Rs_TCO_Grid + Rs_other; % Series resistance in ohm.cm2

shadowing_percentage = finger_width*finger_no*100; % shadowing due to front metal grid

Jsc_CROWM = min(Jsc_top,Jsc_bot) ;
Jsc_total_unshadowed = Jsc_top + Jsc_bot;
Jsc_total_shadowed = Jsc_total_unshadowed*(1 - shadowing_percentage/100);
Jsc_top = Jsc_top*(1 - shadowing_percentage/100);
Jsc_bot = Jsc_bot*(1 - shadowing_percentage/100);% Jsc with shadowing

% Diode model of the bot cell
J01_bot = 110e-15;
n1_bot = 1.1;
J02_bot = 0;
n2_bot = 2;
J0H_bot = 0.1e-6;
nH_bot = 2;
RH_bot = 0.3e3;
Rs_bot = 0;
Rsh_bot = 100.6e3;

% Diode model of the top cell
J01_top = 1e-18;
n1_top = 1.2;
J02_top = 0;
n2_top = 2;
J0H_top = 0.1e-6;
nH_top = 3;
RH_top = 0.2e3;
Rs_top = Rs;
Rsh_top = 20e3;

% Calculations

%%% Calculate Jsc
V = 0;
eqn1 = - J - Jsc_bot + J01_bot*(exp((V_bot - J*Rs_bot)/(n1_bot*Vth)) - 1) + J02_bot*(exp((V_bot - J*Rs_bot)/(n2_bot*Vth)) - 1) + JH_bot + (V_bot - J*Rs_bot)/Rsh_bot == 0;
eqn2 = - JH_bot + J0H_bot*(exp((V_bot - J*Rs_bot - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
eqn3 = - J - Jsc_top + J01_top*(exp(((V - V_bot) - J*Rs_top)/(n1_top*Vth)) - 1) + J02_top*(exp(((V - V_bot) - J*Rs_top)/(n2_top*Vth)) - 1) + JH_top + ((V - V_bot) - J*Rs_top)/Rsh_top == 0;
eqn4 = - JH_top + J0H_top*(exp(((V - V_bot) - J*Rs_top - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
sol = solve([eqn1 eqn2 eqn3 eqn4], [J JH_bot JH_top V_bot]);
Jsc_device = double(sol.J);

% Calculate Voc
eqn1_Voc = - 0  - Jsc_bot + J01_bot*(exp((Voc_var)/(n1_bot*Vth)) - 1) + J02_bot*(exp((Voc_var)/(n2_bot*Vth)) - 1) + JH_bot + (Voc_var)/Rsh_bot == 0;
eqn2_Voc = - JH_bot + J0H_bot*(exp((Voc_var - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
sol_Voc_bot = solve([eqn1_Voc eqn2_Voc], [Voc_var JH_bot]);
Voc_bot = double(sol_Voc_bot.Voc_var);

eqn3_Voc = - 0  - Jsc_top + J01_top*(exp((Voc_var)/(n1_top*Vth)) - 1) + J02_top*(exp((Voc_var)/(n2_top*Vth)) - 1) + JH_top + (Voc_var)/Rsh_top == 0;
eqn4_Voc = - JH_top + J0H_top*(exp((Voc_var - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
sol_Voc_top = solve([eqn3_Voc eqn4_Voc], [Voc_var JH_top]);
Voc_top = double(sol_Voc_top.Voc_var);

Voc_device = Voc_bot + Voc_top;

% Calculate mpp with golden section search algorithm
a = 0;                                     % start of interval
b = Voc_device;                            % end of interval

V_mpp_1 = a + (1 - tau)*(b - a);
V_mpp_2 = a + tau*(b - a);

V = V_mpp_1;
eqn1 = - J - Jsc_bot + J01_bot*(exp((V_bot - J*Rs_bot)/(n1_bot*Vth)) - 1) + J02_bot*(exp((V_bot - J*Rs_bot)/(n2_bot*Vth)) - 1) + JH_bot + (V_bot - J*Rs_bot)/Rsh_bot == 0;
eqn2 = - JH_bot + J0H_bot*(exp((V_bot - J*Rs_bot - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
eqn3 = - J - Jsc_top + J01_top*(exp(((V - V_bot) - J*Rs_top)/(n1_top*Vth)) - 1) + J02_top*(exp(((V - V_bot) - J*Rs_top)/(n2_top*Vth)) - 1) + JH_top + ((V - V_bot) - J*Rs_top)/Rsh_top == 0;
eqn4 = - JH_top + J0H_top*(exp(((V - V_bot) - J*Rs_top - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
sol = solve([eqn1 eqn2 eqn3 eqn4], [J JH_bot JH_top V_bot]);
j_mpp_1 = double(sol.J);
p_mpp_1 = j_mpp_1*V_mpp_1;
V_mpp_hold(1,1) = V_mpp_1;
j_mpp_hold(1,1) = j_mpp_1;
p_mpp_hold(1,1) = p_mpp_1;

V = V_mpp_2;
eqn1 = - J - Jsc_bot + J01_bot*(exp((V_bot - J*Rs_bot)/(n1_bot*Vth)) - 1) + J02_bot*(exp((V_bot - J*Rs_bot)/(n2_bot*Vth)) - 1) + JH_bot + (V_bot - J*Rs_bot)/Rsh_bot == 0;
eqn2 = - JH_bot + J0H_bot*(exp((V_bot - J*Rs_bot - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
eqn3 = - J - Jsc_top + J01_top*(exp(((V - V_bot) - J*Rs_top)/(n1_top*Vth)) - 1) + J02_top*(exp(((V - V_bot) - J*Rs_top)/(n2_top*Vth)) - 1) + JH_top + ((V - V_bot) - J*Rs_top)/Rsh_top == 0;
eqn4 = - JH_top + J0H_top*(exp(((V - V_bot) - J*Rs_top - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
sol = solve([eqn1 eqn2 eqn3 eqn4], [J JH_bot JH_top V_bot]);
j_mpp_2 = double(sol.J);
p_mpp_2 = j_mpp_2*V_mpp_2;

V_mpp_hold(2,1) = V_mpp_2;
j_mpp_hold(2,1) = j_mpp_2;
p_mpp_hold(2,1) = p_mpp_2;

while ((abs(b - a) > epsilon) && (i < iter))

    i = i + 1;

    if(abs(p_mpp_1) > abs(p_mpp_2))
        b = V_mpp_2;
        V_mpp_2 = V_mpp_1;
        V_mpp_1 = a + (1 - tau)*(b - a);
        k_1 = 1; k_2 = 0;
    else
        a = V_mpp_1;
        V_mpp_1 = V_mpp_2;
        V_mpp_2 = a + tau*(b - a);
        k_1 = 0; k_2 = 1;
    end
        V = V_mpp_1;
        eqn1 = - J - Jsc_bot + J01_bot*(exp((V_bot - J*Rs_bot)/(n1_bot*Vth)) - 1) + J02_bot*(exp((V_bot - J*Rs_bot)/(n2_bot*Vth)) - 1) + JH_bot + (V_bot - J*Rs_bot)/Rsh_bot == 0;
        eqn2 = - JH_bot + J0H_bot*(exp((V_bot - J*Rs_bot - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
        eqn3 = - J - Jsc_top + J01_top*(exp(((V - V_bot) - J*Rs_top)/(n1_top*Vth)) - 1) + J02_top*(exp(((V - V_bot) - J*Rs_top)/(n2_top*Vth)) - 1) + JH_top + ((V - V_bot) - J*Rs_top)/Rsh_top == 0;
        eqn4 = - JH_top + J0H_top*(exp(((V - V_bot) - J*Rs_top - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
        sol = solve([eqn1 eqn2 eqn3 eqn4], [J JH_bot JH_top V_bot]);
        j_mpp_1 = double(sol.J);
        p_mpp_1 = j_mpp_1*V_mpp_1;

        V = V_mpp_2;
        eqn1 = - J - Jsc_bot + J01_bot*(exp((V_bot - J*Rs_bot)/(n1_bot*Vth)) - 1) + J02_bot*(exp((V_bot - J*Rs_bot)/(n2_bot*Vth)) - 1) + JH_bot + (V_bot - J*Rs_bot)/Rsh_bot == 0;
        eqn2 = - JH_bot + J0H_bot*(exp((V_bot - J*Rs_bot - JH_bot*RH_bot)/(nH_bot*Vth)) - 1) == 0;
        eqn3 = - J - Jsc_top + J01_top*(exp(((V - V_bot) - J*Rs_top)/(n1_top*Vth)) - 1) + J02_top*(exp(((V - V_bot) - J*Rs_top)/(n2_top*Vth)) - 1) + JH_top + ((V - V_bot) - J*Rs_top)/Rsh_top == 0;
        eqn4 = - JH_top + J0H_top*(exp(((V - V_bot) - J*Rs_top - JH_top*RH_top)/(nH_top*Vth)) - 1) == 0;
        sol = solve([eqn1 eqn2 eqn3 eqn4], [J JH_bot JH_top V_bot]);
        j_mpp_2 = double(sol.J);
        p_mpp_2 = j_mpp_2*V_mpp_2;

        V_mpp_hold(i + 2,1) = V_mpp_1*k_1 + V_mpp_2*k_2;
        j_mpp_hold(i + 2,1) = j_mpp_1*k_1 + j_mpp_2*k_2;
        p_mpp_hold(i + 2,1) = p_mpp_1*k_1 + p_mpp_2*k_2;
end

% figure(1)
% hold all
% plot(V_mpp_hold, abs(p_mpp_hold),'rx')
%
% figure(2)
% hold all
% plot(V_mpp_hold, j_mpp_hold,'rx')
% xlim([0 Voc_device])
% ylim([Jsc_device 0])

T = table(-Jsc_device*1e3, Jsc_CROWM *1e3, Jsc_total_unshadowed*1e3, Jsc_total_shadowed*1e3, Jsc_top*1e3, Jsc_bot*1e3, Voc_device*1e3, p_mpp_hold(end)/(Voc_device*Jsc_device), -p_mpp_hold(end)/1e-3, V_mpp_hold(end)*1e3, -j_mpp_hold(end)*1e3);
Jsc_device=-Jsc_device*1e3;
Jsc_CROWM=Jsc_CROWM *1e3;
Jsc_total_unshadowed=Jsc_total_unshadowed*1e3;
Jsc_total_shadowed=Jsc_total_shadowed*1e3;
Jsc_top=Jsc_top*1e3;
Jsc_bot=Jsc_bot*1e3;
Voc=Voc_device*1e3;
FF=p_mpp_hold(end)/(Voc_device*Jsc_device);
Eff=-p_mpp_hold(end)/1e-3;
V_mpp=V_mpp_hold(end)*1e3;
J_mpp=-j_mpp_hold(end)*1e3;
T.Properties.VariableNames = {'Jsc_device','Jsc_CROWM','Jsc_total_unshadowed','Jsc_total_shadowed','Jsc_top', 'Jsc_bot','Voc','FF','Eff','V_mpp','J_mpp'}
end
