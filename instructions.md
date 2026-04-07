# Instructions For Sharing Zip

## Uploading

On a terminal / command prompt at root level, run the command `rsync -avP "/Users/ma/Documents/GitHub/PokerBots/models.tar.gz" adammotts@login.khoury.northeastern.edu:/tmp/` (but replace the first path with the absolute path to your models.tar.gz and the second with your Khoury Github login)

## Downloading

On a terminal / command prompt at root level, run the command `rsync -avP adammotts@login.khoury.northeastern.edu:/tmp/models.tar.gz "/Users/ma/Documents/GitHub/PokerBots/"` (but replace the first path with your Khoury Github login and the second with the absolute path to the poker project on your computer)