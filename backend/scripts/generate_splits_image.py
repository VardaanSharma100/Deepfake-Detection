import random
import os
def generate_splits(processed_dir,splits_dir,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1):
    os.makedirs(splits_dir,exist_ok=True)
    real_dir=os.path.join(processed_dir,'real')
    fake_dir=os.path.join(processed_dir,'fake')

    real_images=[os.path.join('real',f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    fake_images=[os.path.join('fake',f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    all_data=[(img,0) for img in real_images]+[(img,1)for img in fake_images]
    random.shuffle(all_data)
    
    total=len(all_data)
    train_end=int(train_ratio*total)
    val_end=train_end+int(val_ratio*total)

    train_data=all_data[:train_end]
    val_data=all_data[train_end:val_end]
    test_data=all_data[val_end:]

    def save_split(data,filename):
        with open(os.path.join(splits_dir,filename),'w') as f:
            for path,label in data:
                f.write(f'{path} {label}\n')

    save_split(train_data, 'train.txt')
    save_split(val_data, 'val.txt')
    save_split(test_data, 'test.txt')

if __name__=='__main__':
    generate_splits(processed_dir=r'data/image/processed',splits_dir=r'data/image/splits')


