#python main.py --model VAE --num_samples 1
python main.py --model IWAE --num_samples 50
#python main.py --model IWAE --num_samples 5
#python main.py --model VAE_with_flows --num_samples 1 --need_permute True --num_flows 2 --flow_type RNVP
#python main.py --model VAE_with_flows --num_samples 1 --need_permute False --num_flows 2 --flow_type BNAF
#python main.py --model VAE_with_flows --num_samples 1 --need_permute False --num_flows 2 --flow_type IAF
#python main.py --model VAE_MCMC --num_samples 1