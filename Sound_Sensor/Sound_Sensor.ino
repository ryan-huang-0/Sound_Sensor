/** 
 * Ryan Huang
 * Sound Sensor
 * Adapted from Edge Impulse official example code
 * Located at:
 * '/Arduino/libraries/Sound-detection-project-v6_inferencing/examples/esp32/esp32_microphone_continuous/esp32_microphone_continuous.ino'
 * when Edge Impulse neural network zip library is added to Arduino IDE
 *
 * Information for how to set up IFTTT webhooks can be found at:
 * https://ifttt.com/explore/what-are-webhooks
 *
**/


/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// These sketches are tested with 2.0.4 ESP32 Arduino Core
// https://github.com/espressif/arduino-esp32/releases/tag/2.0.4

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <Sound-detection-project-v6_inferencing.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/i2s.h"

#include <WiFi.h>
#include <HTTPClient.h>
#include "arduino_secrets.h"
#include <stdlib.h>
#include <string.h> 

static SemaphoreHandle_t classifier_mutex = NULL;
#define NOTIFICATION_COOLDOWN_MS 5000 // To prevent spamming on detection
static unsigned long last_notification_time = 0;

#define I2S_MIC_SERIAL_CLOCK     GPIO_NUM_5
#define I2S_MIC_LEFT_RIGHT_CLOCK GPIO_NUM_6
#define I2S_MIC_SERIAL_DATA      GPIO_NUM_4


#define REDPIN 19
#define GREENPIN 21 
#define BLUEPIN 20
#define SPEAKER 1

#define NUM_WINDOWS 3 // number of windows to average over
float avg_predictions[EI_CLASSIFIER_LABEL_COUNT] = {0};
float last_predictions[NUM_WINDOWS][EI_CLASSIFIER_LABEL_COUNT] = {{0}};
int current_window = 0;

// WiFi connection 
const char* ssid = SECRET_SSID;
const char* password = SECRET_PASS; 

// IFTTT Webhook
#define IFTTT_URL SECRET_IFTTT_URL


// LED control: Set an LED to a specific color
void set_led_color(int r, int g, int b) {
    digitalWrite(REDPIN, r);
    digitalWrite(GREENPIN, g);
    digitalWrite(BLUEPIN, b);
}

void trigger_physical_feedback(const char* label) {
    int tone_freq = 0;
    // Turn off all LEDs initially
    set_led_color(LOW, LOW, LOW);
    
    if (strcmp(label, "Microwave Beep") == 0) {
        set_led_color(HIGH, LOW, LOW); // RED
        tone_freq = 440;
    }
    else if (strcmp(label, "Door Bell") == 0) {
        set_led_color(LOW, HIGH, LOW); // GREEN
        tone_freq = 880;
    }
    else if (strcmp(label, "Baby Crying") == 0) {
        set_led_color(LOW, LOW, HIGH); // BLUE
        tone_freq = 1320;
    }

    ledcWriteTone(SPEAKER, tone_freq);
    // Turn off LEDs after a very short delay to make the flash noticeable
    vTaskDelay(pdMS_TO_TICKS(200)); 
    set_led_color(LOW, LOW, LOW);
    // Turn off speaker
    ledcWriteTone(SPEAKER, 0);
}

// Wifi setup
void connect_wifi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected.");
}

// RTOS Task for the BLOCKING HTTP call
void notify_ifttt_task(void* pvParameters) {

    const char* label = (const char*)pvParameters;
    // Trigger Speaker/LED
    trigger_physical_feedback(label);
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(IFTTT_URL);
        http.addHeader("Content-Type", "application/json");
        
        // Include detection label in the JSON body
        String jsonPayload = "{\"value1\":\"Detected: " + String(label) + "\"}";
        
        int httpResponseCode = http.POST(jsonPayload);
        
        if (httpResponseCode > 0) {
            Serial.printf("[HTTP] IFTTT notification sent. Code: %d\n", httpResponseCode);
        } else {
            Serial.printf("[HTTP] IFTTT error: %s\n", http.errorToString(httpResponseCode).c_str());
        }
        http.end();
    }
    
    free((void*)label);
    // Delete the task after it finishes its job
    vTaskDelete(NULL);
}

// Wrapper function to spawn task
void trigger_notification_task(const char* label) {
    // We allocate the label string on the heap so the new task can access it safely
    char* label_copy = strdup(label); 

    xTaskCreate(
        notify_ifttt_task,      // Function that implements the task
        "IFTTT_Notify",         // Name of the task
        1024 * 10,              // Stack size (generous for HTTP)
        (void*)label_copy,      // Parameter to be passed to the task (the label)
        1,                      // Priority (low priority is fine)
        NULL                    // Task handle
    );
}


// ======================== Inferencing setup ========================
/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool record_status = true;



void inference_task(void* pvParameters) {
    while (1) {
        bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        continue;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    // Protect calls to Edge Impulse classifier with mutex so only one task an run it at a time
    if (xSemaphoreTake(classifier_mutex, pdMS_TO_TICKS(200)) == pdTRUE) {
        EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
        xSemaphoreGive(classifier_mutex);

        if (r != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", r);
            continue;
        }
    } else {
        // couldn't get mutex fast enough; skip this round
        ei_printf("WARN: classifier busy - skipping\n");
        continue;
    }


// // ======================== Average Detection ========================
    // Save current predictions into rolling buffer
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        last_predictions[current_window][i] = result.classification[i].value;
    }

    // Compute average over last NUM_WINDOWS windows
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float sum = 0;
        for (int w = 0; w < NUM_WINDOWS; w++) {
            sum += last_predictions[w][i];
        }
        avg_predictions[i] = sum / NUM_WINDOWS;
    }

    current_window = (current_window + 1) % NUM_WINDOWS;

    float threshold = 0.5;  // only consider sounds above 50% confidence
    int detected_class = -1;

    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (avg_predictions[i] > threshold) {
            detected_class = i;
            break; // pick the first one above threshold
        }
    }
    if (detected_class != -1) {
            const char* detected_label = ei_classifier_inferencing_categories[detected_class];
            
            // Check if the detected label is NOT "Noise" AND if the cooldown has expired
            if (strcmp(detected_label, "Noise") != 0) {
                
                // Get current time using FreeRTOS tick count converted to milliseconds
                unsigned long current_time = pdTICKS_TO_MS(xTaskGetTickCount());
                
                // Check if enough time has passed since the last notification
                if (current_time - last_notification_time > NOTIFICATION_COOLDOWN_MS) {
                    
                    ei_printf("Detected: %s (%.2f) - Sending NOTIFICATION!\n", 
                            detected_label, 
                            avg_predictions[detected_class]);
                    

                    
                    
                    // Blocking WIFI Notification (Runs concurrently in new task) + Speaker/LED physical notification
                    trigger_notification_task(detected_label);
                    
                    // Reset the notification time
                    last_notification_time = current_time;
                } else {
                    ei_printf("Detected: %s (%.2f) - In cooldown (Next available in %lu ms)\n", 
                            detected_label, 
                            avg_predictions[detected_class],
                            NOTIFICATION_COOLDOWN_MS - (current_time - last_notification_time));
                }
            } else {
                ei_printf("Detected: Noise\n");
            }
        } else {
            ei_printf("Detected: Noise\n");
        }

// // ========================================================================

// ========================= For viewing classification confidence levels by label (debugging) ============================= 
//     if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
//         // print the predictions
//         ei_printf("Predictions ");
//         ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
//             result.timing.dsp, result.timing.classification, result.timing.anomaly);
//         ei_printf(": \n");
//         for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
//             ei_printf("    %s: ", result.classification[ix].label);
//             ei_printf_float(result.classification[ix].value);
//             ei_printf("\n");
//         }
// #if EI_CLASSIFIER_HAS_ANOMALY == 1
//         ei_printf("    anomaly score: ");
//         ei_printf_float(result.anomaly);
//         ei_printf("\n");
// #endif

//         print_results = 0;
//     }

// ========================================================================
    vTaskDelay(1);
    }
}

/**
 * @brief      Arduino setup function
 */
void setup()
{
    Serial.begin(115200);

    classifier_mutex = xSemaphoreCreateMutex();
    if (!classifier_mutex) {
        Serial.println("ERR: couldn't create classifier mutex");
    }

    // ================ Speaker and LED setup ================
    ledcAttach(SPEAKER, 2000, 8);
    
    pinMode(REDPIN, OUTPUT); 
    pinMode(GREENPIN, OUTPUT); 
    pinMode(BLUEPIN, OUTPUT); 
    set_led_color(LOW, LOW, LOW);


    // ================ Wifi & notification setup ================
    connect_wifi();

    // ================ Edge Impulse Setup ================
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Sound Sensor");

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    ei_printf("\nStarting continuous inference in 2 seconds...\n");
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }

    xTaskCreatePinnedToCore(
        inference_task,     // Function that implements the task
        "EI_Inference",     // Name of the task
        1024 * 32,          // Stack size (generous)
        NULL,               // Parameter to the task
        8,                  // Lower priority than capture samples, higher than notify
        NULL,               // Task handle
        tskNO_AFFINITY      // CORE: No Affinity, uses which ever core is not in use
    );

    ei_printf("Recording...\n");
}

/**
 * @brief      Arduino main function.
 */
void loop()
{
    vTaskDelay(1000);
}

static void audio_inference_callback(uint32_t n_bytes)
{
    for(int i = 0; i < n_bytes>>1; i++) {
        inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

        if(inference.buf_count >= inference.n_samples) {
            inference.buf_select ^= 1;
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {

  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status) {

    /* read data at once from i2s */
    i2s_read(I2S_NUM_0, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

    if (bytes_read <= 0) {
      ei_printf("Error in I2S read : %d", bytes_read);
    }
    else {
        if (bytes_read < i2s_bytes_to_read) {
        ei_printf("Partial I2S read");
        }

        // scale the data (otherwise the sound is too quiet)
        for (int x = 0; x < i2s_bytes_to_read/2; x++) {
            sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 5;
            // int32_t sample = (int32_t)sampleBuffer[x];
            // sampleBuffer[x] = (int16_t)(sample << 4);
        }

        if (record_status) {
            audio_inference_callback(i2s_bytes_to_read);
        }
        else {
            break;
        }
    }
  }
  vTaskDelete(NULL);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        ei_free(inference.buffers[0]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    if (i2s_init(EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start I2S!");
    }

    ei_sleep(100);

    record_status = true;

    // xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, NULL);

    xTaskCreatePinnedToCore(
        capture_samples, 
        "CaptureSamples", 
        1024 * 32, 
        (void*)sample_buffer_size, 
        10,  // high priority
        NULL, 
        0 // Core ID 0
    );

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    i2s_deinit();
    ei_free(inference.buffers[0]);
    ei_free(inference.buffers[1]);
}

// ================== I2S Microphone Settings ==================
static int i2s_init(uint32_t sampling_rate) {
  // Start listening for audio: MONO @ 8/16KHz
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
      .sample_rate = sampling_rate,
      .bits_per_sample = (i2s_bits_per_sample_t)16,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 8,
      .dma_buf_len = 512,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };
  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_MIC_SERIAL_CLOCK,    // IIS_SCLK
      .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,     // IIS_LCLK
      .data_out_num = I2S_PIN_NO_CHANGE,  // IIS_DSIN
      .data_in_num = I2S_MIC_SERIAL_DATA   // IIS_DOUT
  };
  esp_err_t ret = 0;

  ret = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ei_printf("Error in i2s_driver_install");
  }

  ret = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (ret != ESP_OK) {
    ei_printf("Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer(I2S_NUM_0);
  if (ret != ESP_OK) {
    ei_printf("Error in initializing dma buffer with 0");
  }

  return int(ret);
}

static int i2s_deinit(void) {
    i2s_driver_uninstall(I2S_NUM_0); //stop & destroy i2s driver
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif