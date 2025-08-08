import torch  # pytorch kütüphanesi, tensor işlemleri
import torch.nn as nn  # yapay zeka sinir ağları kullanmak için
import torch.optim as optim  # optimizasyon algoritmaları içeren modül (düzeltildi: optin → optim)
import torchvision  # görüntü işleme ve pre-defined modelleri içerir
import torchvision.transforms as transforms  # görüntü dönüşümleri yapmak için
import matplotlib.pyplot as plt  # temel görselleştirme


# veri seti yükleme - data klasörüne yerleştirdik
def get_data_loaders(batch_size=64):  # her iterasyonda işlenecek veri miktarı
    transform = transforms.Compose([
        transforms.ToTensor(),  # görüntüyü tensore çevirir ve 0-255  -> 0-1 ölçeklendirir
        transforms.Normalize((0.5,), (0.5,))  # pixel değerlerini  -1 ile 1 arasına ölçekler
    ])

    # mnist veri seti yükleme
    train_set = torchvision.datasets.MNIST(root="~/data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="~/data", train=False, download=True, transform=transform)

    # pytorch veri yükleyicisi
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Veri yükleyicileri al
train_loader, test_loader = get_data_loaders()


# Data visualization
def visualize_samples(loader, n):
    images, labels = next(iter(loader))  # ilk batchten görüntü ve etiket alma işlemi
    print(images[0].shape)
    fig, axes = plt.subplots(1, n, figsize=(10, 5))  # n farklı görüntü için görselleştirme alanı

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")  # squeeze() ile boyutu [1,28,28] → [28,28] yapılır
        
        axes[i].set_title(f"Label: {labels[i].item()}") #görüntüye ait sınıf etiketini başlık olarak yaz.
        axes[i].axis("off") #eksenleri gizledik.
    plt.show()
visualize_samples(train_loader, 4)

#define ann model

# yapay sinir ağı class
class NeuralNetwork(nn.Module):  #pytorchun nn.Modelu den miras aldı

    #nn inşa etmek için gerekli olan bileşenleri tanımladık.
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        
        # 2 boyutlu görüntüleri vektör haline çeviriyoruz (1D hale)
        self.flatten=nn.Flatten()
           
        # ilk tam bağlı katmanı oluştur.
        self.fc1=nn.Linear(28*28, 128) #784 ->input_size ,128->output size
        
        # aktivasyon katmanını oluştur.
        self.relu=nn.ReLU()
        
        #ikinci tam bağlı katmanı oluştur.
        self.fc2=nn.Linear(128,64) #input yine 128 olur genelde yarıya düşerek output size verir.
        
        #çıktı katmanını oluştur
        self.fc3=nn.Linear(64,10 ) #fc2 çünkü 64 çıktı vermişti. input değeri bu yüzden 64 olur. Output size 10 olmak zorunda (10 sınıflı sınıflandırma problemi çözüyoruz çünkü.)
        
        
        
        
        
        
    def forward(self,x): #ileri yayılım. Giriş olarak x alsın. (görüntü)
        x=self.flatten(x)
        # initial x=28*28'lik bir görüntüdür. -> düzleşitrip 784 vektör hakşne getireceğiz.
        x=self.fc1(x) #birinci bağlı katman
        x=self.relu(x) #aktivasyo katmanı
        x=self.fc2(x) #ikinci bağlı katman
        x=self.relu(x) #aktivasyo katmanı
        x=self.fc3(x) #output katmanı
        
        #model çıktısını return ettik
        return x
        
# Modeli oluştur ve device'a taşı
device = torch.device("cpu")

model = NeuralNetwork().to(device)

# Kayıp fonksiyonu ve optimizasyon algoritmasını tanımla
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),          # Çok sınıflı sınıflandırma için kayıp fonksiyonu
    optim.Adam(model.parameters(), lr=0.001)  # Adam optimizasyon algoritması, öğrenme oranı 0.001
)
criterion,optimizer=define_loss_and_optimizer(model)

# Kayıp fonksiyonu ve optimizer değişkenlerini hazırla
loss_fn, optimizer = define_loss_and_optimizer(model)

#train

def train_model(train,train_loader,criterion,optimizer,epochs=10):
    #modeli eğitim moduna alalım
    model.train()
        
    #her bir epochs sonunda elde edilen kayıp los değerleri saklamak için liste tanımla.
    train_losses=[]
    
    #belirtilen epoch sayısı kadar eğitim yapalım.
    for epoch in range(epochs):
        total_loss=0 #toplam kayıp değeri
        
        #tüm eğitim verileri üzerinde iterasyon gerçekleştir.
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device) #verileri cihaza taşıdık
    
            #gradyanları sıfırla
            optimizer.zero_grad() 
    
            #modeli uygyla (forward pro)
            predictions=model(images)
    
            #los hesaplama ->y prediction şle y_real
            loss=criterion(predictions,labels) #doğru hesaplandı mı hesaplanamdı mı
    
            #geri yayılım yani gradyan hesaplama 
            loss.backward()
            #update weight -> ağırlıkları güncelleme
            optimizer.step()
            
            total_loss=total_loss+loss.item()
        #ortalama kayıp hesaplama    
        avg_loss=total_loss/len(train_loader)
        train_losses.append(avg_loss) #listeye ekledik.
        print(f"Epoch {epoch+1}/{epochs}, Loss:{avg_loss:.3f}")
    #loss graph
    
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel(" Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
train_model(model, train_loader, criterion, optimizer,epochs=5) 

#test

def test_model(model,test_loader):
    #modeli değerlendirme moduna aldık
    model.eval()
    
    #doğru tahmin sayacı
    correct=0
    
    #toplam veri sayısı sayacı
    total=0
    
    #gradyan hesaplama gereksiz, kaptıldı.
    with torch.no_grad():
        for images,labels in test_loader: #test veri setini döngüye aldık.
            images,labels=images.to(device),labels.to(device) #verileri cihaza taşı.
            predictions=model(images)
            
            #en yüksek olasılıklı sınıfın etiketini bul
            _,predicted=torch.max(predictions,1)
            total+=labels.size(0)
            
            #doğru tahminleri say. Listeyi item olarka iterasyon edip üzerine saydırdık
            correct+=(predicted==labels).sum().item() 
    print(f"Test Accuracy: {100*correct/total:.3f}%")
test_model(model, test_loader)

# ANA PROGRAM

if __name__=="__main__":
    #veri yükleyicilerini al
    train_loader,test_loader=get_data_loaders()
    visualize_samples(train_loader,5)
    model=NeuralNetwork().to(device)
    criterion,optimizer=define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
