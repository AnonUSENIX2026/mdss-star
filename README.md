# Secure Aggregation

This repo contains a number of programs and scripts related to the MDSS-STAR project.

## How To Test the Decoder

1. Install the [NTL](https://libntl.org/) and [JSON](https://github.com/nlohmann/json) libraries.
1. `cd decoder`
1. `mkdir build`
1. `cd build`
1. `cmake ..`
1. `make`
1. `./ch_decoder ../example_instances/tiny.json ../configs/tiny.json`

## How to Run Remote Benchmarks

1. Generate a configuration file `config.json` using `/parameter_calculator/main.sage`.
1. Generate an instance file `instance.json` using `/instance_generation/instance_generation.sage`.
1. Place both of the above files in `/infrastructure/ansible`.
1. Enter the `/infrastructure/pulumi` directory.
1. Edit `__main__.py` and set the `results_bucket.bucket` property.
1. Create an `ssh_cidrs.json` file with the following structure:
   ```json
   [
     {
       "resource_name": "cidr_name",
       "cidr": "0.0.0.0/32",
       "name": "Longer name",
       "description": "Description"
     }
   ]
   ```
   with an entry for each address you would like to ssh to your server from.
1. Initialize [Pulumi](https://www.pulumi.com/) and build the stack.
1. Enter the `/infrastructure/ansible` directory.
1. Install [Ansible](https://docs.ansible.com/ansible/latest/index.html)
1. Fill in the TODOs in the ansible files
1. Run `ansible-playbook -i aws-ec2.yml install-dependencies.yml`
1. Run `ansible-playbook -i aws-ec2.yml benchmark-decoder.yml`
1. View results in `s3`.
