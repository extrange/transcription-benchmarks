#/usr/bin/env bash

sudo apt autoremove --purge
sudo apt autoclean
sudo docker system prune -a
sudo docker volume prune
uv cache clean
