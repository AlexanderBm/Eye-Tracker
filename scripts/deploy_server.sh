#!/usr/bin/expect -f
# Script to deploy server files to Raspberry Pi
# Usage: ./deploy_server.sh <RPI_IP> <RPI_USER> <RPI_PASSWORD>

set timeout 30

if {[llength $argv] < 3} {
    puts "Usage: ./deploy_server.sh <RPI_IP> <RPI_USER> <RPI_PASSWORD>"
    exit 1
}

set rpi_ip [lindex $argv 0]
set rpi_user [lindex $argv 1]
set rpi_pass [lindex $argv 2]

# Get the directory where the script is located
set script_dir [file dirname [info script]]

puts "Deploying to $rpi_user@$rpi_ip..."

# 1. Copy start script
spawn scp $script_dir/start_stream_sw.sh $rpi_user@$rpi_ip:~/start_stream_sw.sh
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { send "$rpi_pass\r" }
}
expect eof

# 2. Copy setup script
spawn scp $script_dir/setup_server.sh $rpi_user@$rpi_ip:~/setup_server.sh
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { send "$rpi_pass\r" }
}
expect eof

# 3. Make scripts executable
spawn ssh $rpi_user@$rpi_ip "chmod +x ~/start_stream_sw.sh ~/setup_server.sh"
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { send "$rpi_pass\r" }
}
expect eof

puts "Deployment complete!"
puts "To install dependencies on RPi, run: ssh $rpi_user@$rpi_ip './setup_server.sh'"
