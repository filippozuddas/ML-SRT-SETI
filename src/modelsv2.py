import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) per campionare il vettore latente z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape=(16, 512, 1), latent_dim=8):
    """
    Costruisce l'Encoder con 8 layer convoluzionali come da paper.
    """
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.BatchNormalization()(encoder_inputs)
    
    # Architettura profonda (8 layer conv)
    filters = [32, 32, 64, 64, 128, 128, 128, 128]
    
    for f in filters:
        x = layers.Conv2D(f, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim=8, target_shape=(16, 512, 1)):
    """
    Costruisce il Decoder speculare all'encoder per ricostruire lo spettrogramma.
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    # Dimensioni di partenza per l'upsampling (devono matchare l'uscita piatta dell'encoder)
    # Con 8 strati di stride 2, l'input viene ridotto di un fattore 2^8 = 256.
    # 512 / 256 = 2 pixel di larghezza. 16 / 256 < 1 pixel (problema!).
    # Il paper usa stride=1 in alcuni layer per evitare di ridurre troppo.
    # Qui ricostruiamo partendo da una base solida e upsamplando.
    
    # Partiamo da 1x32 con 128 filtri
    x = layers.Dense(1 * 32 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((1, 32, 128))(x)
    x = layers.BatchNormalization()(x)
    
    # Upsampling progressivo per arrivare a 16x512
    # H: 1 -> 2 -> 4 -> 8 -> 16 (4 upsamples)
    # W: 32 -> 64 -> 128 -> 256 -> 512 (4 upsamples)
    
    filters_rev = [128, 64, 32, 32]
    
    for f in filters_rev:
        x = layers.Conv2DTranspose(f, kernel_size=(3, 3), strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
    decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(3, 3), activation="sigmoid", padding="same")(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class SETI_VAE(keras.Model):
    def __init__(self, input_shape=(16, 512, 1), latent_dim=8, beta=1.5, alpha=10.0, **kwargs):
        super(SETI_VAE, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.beta = beta   # Peso KL Divergence (Paper: 1.5)
        self.alpha = alpha # Peso Clustering Loss (Paper: 10.0)
        
        self.encoder = build_encoder(input_shape, latent_dim)
        self.decoder = build_decoder(latent_dim, input_shape)
        
        # Metrics Trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cluster_loss_tracker = keras.metrics.Mean(name="cluster_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, 
                self.kl_loss_tracker, self.cluster_loss_tracker]

    def call(self, inputs):
        # Gestisce sia input singoli (16,512) che cadenze (6,16,512)
        if len(inputs.shape) == 5: # (Batch, 6, 16, 512, 1)
            batch_size = tf.shape(inputs)[0]
            # Appiattiamo la dimensione temporale nel batch per passare nell'encoder
            reshaped = tf.reshape(inputs, (-1, 16, 512, 1))
            z_mean, z_log_var, z = self.encoder(reshaped)
            reconstruction = self.decoder(z)
            
            # Ripristiniamo la forma della cadenza
            reconstruction = tf.reshape(reconstruction, (batch_size, 6, 16, 512, 1))
            z = tf.reshape(z, (batch_size, 6, self.latent_dim))
            z_mean = tf.reshape(z_mean, (batch_size, 6, self.latent_dim))
            z_log_var = tf.reshape(z_log_var, (batch_size, 6, self.latent_dim))
            
            return reconstruction, z_mean, z_log_var, z
        else:
            return self.encoder(inputs)

    def clustering_loss(self, z, labels):
        """
        Calcola la loss di clustering (Eq 4 e 5 del paper).
        z: (Batch, 6, LatentDim)
        labels: (Batch, ) - 1=SETI, 0=RFI/Noise
        """
        # Separiamo ON (0,2,4) e OFF (1,3,5)
        ons = tf.gather(z, [0, 2, 4], axis=1)  # (Batch, 3, 8)
        offs = tf.gather(z, [1, 3, 5], axis=1) # (Batch, 3, 8)
        
        # --- SETI LOGIC (Label = 1) ---
        # Gli ON devono essere simili tra loro (minimizzare varianza)
        centroid_on = tf.reduce_mean(ons, axis=1, keepdims=True)
        dist_on_on = tf.reduce_mean(tf.reduce_sum(tf.square(ons - centroid_on), axis=2), axis=1)
        
        # Gli OFF devono essere simili tra loro
        centroid_off = tf.reduce_mean(offs, axis=1, keepdims=True)
        dist_off_off = tf.reduce_mean(tf.reduce_sum(tf.square(offs - centroid_off), axis=2), axis=1)
        
        # ON e OFF devono essere DIVERSI (massimizzare distanza)
        # Usiamo exp(-distanza) per penalizzare se sono vicini
        dist_on_off = tf.reduce_sum(tf.square(centroid_on - centroid_off), axis=2) # (Batch, 1)
        loss_push = tf.exp(-dist_on_off) 
        
        loss_seti = dist_on_on + dist_off_off + tf.squeeze(loss_push)
        
        # --- RFI LOGIC (Label = 0) ---
        # Tutto deve essere simile a tutto (tutti i 6 pannelli vicini al centroide globale)
        centroid_all = tf.reduce_mean(z, axis=1, keepdims=True)
        loss_rfi = tf.reduce_mean(tf.reduce_sum(tf.square(z - centroid_all), axis=2), axis=1)
        
        # Seleziona la loss giusta in base alla label del batch
        labels = tf.cast(labels, tf.float32)
        final_cluster_loss = labels * loss_seti + (1 - labels) * loss_rfi
        
        return tf.reduce_mean(final_cluster_loss)

    def train_step(self, data):
        # Unpack dei dati dal generatore (X, y)
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            y = None

        with tf.GradientTape() as tape:
            # Forward pass
            reconstruction, z_mean, z_log_var, z = self(x)
            
            # 1. Reconstruction Loss
            # Somma su pixel, media su batch
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, reconstruction), axis=(2, 3)
                )
            )
            
            # 2. KL Divergence Loss (Regolarizzazione spazio latente)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=2)) # Media su batch e cadenza
            
            # 3. Clustering Loss (Solo se abbiamo le label)
            if y is not None:
                cluster_loss = self.clustering_loss(z, y)
            else:
                cluster_loss = 0.0
            
            # Somma pesata finale (Eq 6)
            total_loss = recon_loss + (self.beta * kl_loss) + (self.alpha * cluster_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.cluster_loss_tracker.update_state(cluster_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cluster_loss": self.cluster_loss_tracker.result(),
        }
        
        
if __name__ == "__main__":
    # Configurazione
    input_shape = (6, 16, 512, 1) # (Batch, Cadence, Time, Freq, Channel)
    model = SETI_VAE()

    # Creiamo un input finto
    dummy_input = tf.random.normal((2, 6, 16, 512, 1))
    dummy_labels = tf.constant([1, 0]) # Un SETI, un RFI

    # Test chiamata (Build lazy)
    print("Esecuzione chiamata di test...")
    outputs = model(dummy_input)
    recon = outputs[0]

    print(f"Shape Output Ricostruito: {recon.shape}")
    # Deve essere (2, 6, 16, 512, 1)

    # Test Training Step (verifica gradienti)
    print("Test Training Step...")
    model.compile(optimizer='adam')
    history = model.train_on_batch(dummy_input, dummy_labels)
    print("Loss calcolata:", history)
    print("✅ Modello costruito e compilabile!")