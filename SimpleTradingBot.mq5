#property copyright "My Trading Bot"
#property link      "https://example.com"
#property version   "1.00"

#include <Trade\Trade.mqh> // Library for trading functions
CTrade trade; // Object to handle trades

input string SignalFile = "trade_signal.txt"; // File to read signals from

int OnInit() {
   Print("SimpleTradingBot initialized on ", Symbol(), ", ", Period());
   // Get the full file path
   string file_path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + SignalFile;
   Print("Signal file path (local): ", file_path);
   file_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\" + SignalFile;
   Print("Signal file path (common): ", file_path);
   // Test reading the signal file during initialization
   string signal = ReadSignal();
   Print("Initial signal check: ", signal);
   // If the file doesn't exist, create it with a default signal
   if (signal == "") {
      Print("Signal file empty or not found, creating with default signal: BUY");
      WriteSignal("BUY");
      signal = ReadSignal();
      Print("Signal after creating file: ", signal);
   }
   return(INIT_SUCCEEDED);
}

void OnTick() {
   // Check for a new signal every tick (price update)
   string signal = ReadSignal();
   
   // Log the signal
   Print("Signal read: ", signal);
   
   // Handle both English and Chinese signals
   if (signal == "BUY" || signal == "购买") {
      trade.Buy(0.1); // Buy 0.1 lots
      Print("Buy order placed");
      WriteSignal(""); // Clear signal after execution
   }
   else if (signal == "SELL" || signal == "卖出") {
      trade.Sell(0.1); // Sell 0.1 lots
      Print("Sell order placed");
      WriteSignal(""); // Clear signal after execution
   }
}

// Function to read signal from file
string ReadSignal() {
   string signal = "";
   // Try the common folder first
   int file_handle = FileOpen(SignalFile, FILE_READ | FILE_TXT | FILE_COMMON);
   if (file_handle != INVALID_HANDLE) {
      signal = FileReadString(file_handle);
      FileClose(file_handle);
      Print("Successfully read signal from common folder: ", signal);
   }
   else {
      Print("Failed to open signal file in common folder: ", GetLastError());
      // Fallback to local folder
      file_handle = FileOpen(SignalFile, FILE_READ | FILE_TXT);
      if (file_handle != INVALID_HANDLE) {
         signal = FileReadString(file_handle);
         FileClose(file_handle);
         Print("Successfully read signal from local folder: ", signal);
      }
      else {
         Print("Failed to open signal file in local folder: ", GetLastError());
      }
   }
   return signal;
}

// Function to write (or clear) signal to file
void WriteSignal(string text) {
   // Try the common folder first
   int file_handle = FileOpen(SignalFile, FILE_WRITE | FILE_TXT | FILE_COMMON);
   if (file_handle != INVALID_HANDLE) {
      FileWriteString(file_handle, text);
      FileClose(file_handle);
      Print("Successfully wrote signal to common folder: ", text);
   }
   else {
      Print("Failed to write signal file in common folder: ", GetLastError());
      // Fallback to local folder
      file_handle = FileOpen(SignalFile, FILE_WRITE | FILE_TXT);
      if (file_handle != INVALID_HANDLE) {
         FileWriteString(file_handle, text);
         FileClose(file_handle);
         Print("Successfully wrote signal to local folder: ", text);
      }
      else {
         Print("Failed to write signal file in local folder: ", GetLastError());
      }
   }
}