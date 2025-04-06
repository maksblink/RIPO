import cv2
import time
import logging
import os
import numpy as np
from datetime import datetime
from typing import Union


class ObjectDetectionSystem:
    def __init__(self):
        self.logger = self.setup_logger()
        self.running = False
        self.detection_enabled = False
        self.collection_enabled = False
        self.source = None
        self.cap = None

        self.last_saved_time = 0
        self.save_interval = 3  # sekundy
        self.output_folder = "wzorce"
        os.makedirs(self.output_folder, exist_ok=True)

    def setup_logger(self):
        logging.basicConfig(
            filename='object_detection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()

    def start_video_source(self, source: Union[int, str]):
        try:
            if isinstance(source, str) and source.startswith('"') and source.endswith('"'):
                source = source[1:-1]

            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Nie można otworzyć źródła wideo: {source}")

            self.source = source
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Uruchomiono źródło: {source}, Rozdzielczość: {width}x{height}, FPS: {fps}")
            print(f"\nUruchomiono wideo: {os.path.basename(str(source))}")
            print(f"Rozdzielczość: {width}x{height}, FPS: {fps:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Błąd przy uruchamianiu źródła: {str(e)}")
            print(f"\nBŁĄD: {str(e)}")
            return False

    def stop_video_source(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.source = None
            self.logger.info("Zatrzymano źródło wideo")

    def start_collection_mode(self):
        self.collection_enabled = True
        self.detection_enabled = False
        self.last_saved_time = 0  # resetujemy czas
        self.logger.info("Uruchomiono tryb zbierania zdjęć wzorcowych")
        print("Aktywny tryb: ZBIERANIE WZORCÓW")

    def start_detection_mode(self):
        self.detection_enabled = True
        self.collection_enabled = False
        self.logger.info("Uruchomiono tryb rozpoznawania obiektów")
        print("Aktywny tryb: ROZPOZNAWANIE")

    def stop_processing(self):
        self.collection_enabled = False
        self.detection_enabled = False
        self.logger.info("Zatrzymano przetwarzanie")
        print("Tryb przetwarzania: WYŁĄCZONY")

    def process_frame(self, frame):
        processed_frame = frame.copy()
        cv2.putText(processed_frame, f"Czas: {datetime.now().strftime('%H:%M:%S')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.collection_enabled:
            cv2.putText(processed_frame, "TRYB ZBIERANIA WZORCOW",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            current_time = time.time()
            if current_time - self.last_saved_time >= self.save_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_folder, f"wzorzec_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                self.last_saved_time = current_time
                self.logger.info(f"Zapisano wzorzec: {filename}")
                print(f"[ZBIERANIE] Zapisano wzorzec → {filename}")

        elif self.detection_enabled:
            cv2.putText(processed_frame, "TRYB ROZPOZNAWANIA",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(processed_frame, "Obiekt", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            self.logger.info(f"Wykryto {len(contours)} konturów o powierzchni >500px")

        return processed_frame

    def run(self):
        self.running = True
        frame_count = 0
        start_time = time.time()

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("\nKoniec wideo lub błąd odczytu klatki")
                self.logger.error("Błąd odczytu klatki lub koniec strumienia")
                break

            processed_frame = self.process_frame(frame)
            frame_count += 1

            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('System detekcji obiektów', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_collection_mode()
            elif key == ord('d'):
                self.start_detection_mode()
            elif key == ord('s'):
                self.stop_processing()
            elif key == ord('p'):
                while True:
                    key_pause = cv2.waitKey(1)
                    if key_pause == ord('p'):
                        break
                    elif key_pause == ord('q'):
                        self.running = False
                        break

        self.stop_video_source()
        cv2.destroyAllWindows()
        self.running = False
        print("Program zakończony")


def main():
    system = ObjectDetectionSystem()

    print("\n" + "=" * 50)
    print("SYSTEM DETEKCJI OBIEKTÓW")
    print("=" * 50 + "\n")

    print("Wybierz źródło wideo:")
    print("1 - Kamera lokalna")
    print("2 - Plik wideo")
    print("3 - Kamera IP")
    print("4 - Wyjście")

    while True:
        choice = input("\nTwój wybór (1-4): ")

        if choice == '1':
            source = 0
            break
        elif choice == '2':
            while True:
                file_path = input("\nPodaj ścieżkę do pliku wideo: ").strip()
                if file_path.startswith('"') and file_path.endswith('"'):
                    file_path = file_path[1:-1]

                if not os.path.exists(file_path):
                    print("BŁĄD: Plik nie istnieje! Spróbuj ponownie.")
                    continue

                if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                    print("UWAGA: To może nie być obsługiwany format wideo. Spróbuj pliku .mp4")

                source = file_path
                break
            break
        elif choice == '3':
            ip_camera_url = input("\nPodaj URL kamery IP (np. rtsp://192.168.1.10/stream): ").strip()
            source = ip_camera_url
            break
        elif choice == '4':
            print("Zamykanie programu...")
            return
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

    if not system.start_video_source(source):
        print("\nNie udało się uruchomić źródła wideo. Sprawdź:")
        print("- Czy kamera jest podłączona (dla wyboru 1)")
        print("- Czy plik wideo istnieje i jest nieuszkodzony (dla wyboru 2)")
        print("- Czy adres IP kamery jest prawidłowy (dla wyboru 3)")
        print("\nSzczegóły błędu zapisano w pliku object_detection.log")
        return

    print("\n" + "=" * 50)
    print("STEROWANIE:")
    print("c - Tryb zbierania wzorców")
    print("d - Tryb rozpoznawania")
    print("s - Stop przetwarzania")
    print("p - Pauza/wznowienie")
    print("q - Wyjście")
    print("=" * 50 + "\n")

    try:
        system.run()
    except Exception as e:
        print(f"\nBŁĄD: {str(e)}")
        system.logger.error(f"Krytyczny błąd: {str(e)}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
