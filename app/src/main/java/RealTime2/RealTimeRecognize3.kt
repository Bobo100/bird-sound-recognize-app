package RealTime2

import JLibrosa.JLibrosa
import JLibrosa.process.AudioFeatureExtraction
import RealTime2.aed_method_DSP.FIR
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.example.mysoundtest.R
import kotlin.math.roundToInt


@Suppress("DEPRECATION")
class RealTimeRecognize3 : AppCompatActivity() {
    companion object {
        private const val REQUEST_RECORD_AUDIO = 1
    }

    private var mAudioRecord: AudioRecord? = null

    // The thread allowed to keep recording; null when stopped
    @Volatile
    private var recordingThread: Thread? = null

    private var laiyuan = MediaRecorder.AudioSource.VOICE_RECOGNITION //来源
    private val SAMPLE_RATE = 44100 //采样频率
    private val SAMPLE_DURATION_MS = 1000
    private val CHANNEL = AudioFormat.CHANNEL_IN_MONO//声道
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT //格式
    private val MONO_RECORDING_LENGTH = (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000)

    var bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, audioFormat) //缓冲区大小

    private lateinit var mAcousticEventDetector: AcousticEventDetector
    private lateinit var resultView: TextView

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_recognize3)
        resultView = findViewById(R.id.recognize_result)

        //開始錄音 與 辨識
        val start_record = findViewById<Button>(R.id.btn_recognize_start)
        start_record.setOnClickListener {
            if (mAudioRecord == null) {
                startRecord()
            }
        }

        //結束錄音
        val stop_record = findViewById<Button>(R.id.btn_recognize_stop)
        stop_record.setOnClickListener {
            if (mAudioRecord != null) {
                stopRecord()
            }
        }

        // Ask for the microphone up front (not again after a rotation)
        if (savedInstanceState == null && !hasMicrophonePermission()) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }

        // TensorFlow Lite
        mAcousticEventDetector = AcousticEventDetector(
            this,
            Classifier.Device.CPU,
            4,
            melFilterbankSize = 256, //256
            featureWidth = 128 //128
        )
    }

    override fun onStop() {
        super.onStop()
        stopRecord()
    }

    private fun hasMicrophonePermission() =
        ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "未取得麥克風權限，請到系統設定開啟", Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("MissingPermission") // checked by hasMicrophonePermission()
    private fun startRecord() {
        Log.v("start record", "start record")

        if (!hasMicrophonePermission()) {
            Toast.makeText(this, "需要麥克風權限才能辨識", Toast.LENGTH_SHORT).show()
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
            return
        }
        val recorder = AudioRecord(
            laiyuan, SAMPLE_RATE,
            CHANNEL, audioFormat, bufferSize
        )
        recorder.startRecording()
        mAudioRecord = recorder

        val thread = Thread { recordLoop(recorder) }
        recordingThread = thread
        thread.start()
    }

    // Runs on the recording thread until stopRecord() clears recordingThread.
    // The thread owns its recorder, so a quick stop/start never shares one between two threads.
    private fun recordLoop(recorder: AudioRecord) {
        val self = Thread.currentThread()
        val audioBuffer = ShortArray(bufferSize)
        val window = ShortArray(MONO_RECORDING_LENGTH)
        var filled = 0
        try {
            while (recordingThread === self) {
                val read = recorder.read(audioBuffer, 0, audioBuffer.size)
                if (read < 0) {
                    Log.e("record", "AudioRecord.read failed: $read")
                    break
                }

                // Fill the 1 s window; samples past its end are dropped
                val n = minOf(read, window.size - filled)
                System.arraycopy(audioBuffer, 0, window, filled, n)
                filled += n

                if (filled == window.size) {
                    val results = recognize(window)
                    Log.v("results", results.toString())
                    runOnUiThread {
                        // Ignore a window that finished after stop was pressed
                        if (recordingThread === self) resultView.text = "$results"
                    }
                    filled = 0
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
            runOnUiThread { if (mAudioRecord === recorder) mAudioRecord = null }
        }
    }

    // Same feature extraction as utils/audio.melspec in the training repo
    private fun recognize(window: ShortArray): List<Classifier.Recognition> {
        val temp = convertFromShortArrayToFloatArray(window)
        applyPreEmphasis(temp)

        ////////////////////////////////////////////////////////////////////////////////////////// JLibrosa引用

        val jLibrosa = JLibrosa()

        /////////////////////////////////////////////////////////////////////////////////////////// MelSpectroGram
        val melSVal =
            jLibrosa.generateMelSpectroGram(temp, SAMPLE_RATE, 1024, 128, 172)
        /////////////////////////////////////////////////////////////////////////////////////////// amplitudeToDb
        val AudioFeatureExtraction = AudioFeatureExtraction()
        val log_spec = AudioFeatureExtraction.amplitudeToDb(
            convertFrom2DFloatArrayTo2DDoubleArray(melSVal))
        /////////////////////////////////////////////////////////////////////////////////////////// flip_vertically and reshape
        val convertvalue = convertFrom2DDoubleArrayTo2DFloatArray(log_spec)
        val reshape2 = reshape(flip_vertically(convertvalue))
        /////////////////////////////////////////////////////////////////////////////////////////// normalize
        val finalmfccValues = normalize(reshape2)
        /////////////////////////////////////////////////////////////////////////////////////////// 2d to 1d array
        val oneDArray = mode(finalmfccValues)

        /////////////////////////////////////////////////////////////////////////////////////////// predict
        // The interpreter and its input buffer are not thread-safe
        return synchronized(mAcousticEventDetector) { mAcousticEventDetector.recognize(oneDArray) }
    }

    private fun normalize(mfccValues: Array<FloatArray>): Array<IntArray> {
        // Start from a real value: dB values are <= 0, so 0F / 255F never get replaced
        var max_float = mfccValues[0][0]
        var min_float = mfccValues[0][0]
        for (i in mfccValues.indices) {
            for (j in 0 until mfccValues[0].size) {
                if (mfccValues[i][j] > max_float) max_float = mfccValues[i][j]
                if (mfccValues[i][j] < min_float) min_float = mfccValues[i][j]
            }
        }
        val maxMin = max_float - min_float

        val intArray = Array(mfccValues.size) {
            IntArray(mfccValues[0].size)
        }

        for (i in mfccValues.indices) {
            for (j in 0 until mfccValues[0].size) {
                intArray[i][j] = if (maxMin > 0F) ((mfccValues[i][j] - min_float) * 255F / maxMin).roundToInt() else 0
            }
        }
        return intArray
    }

    // Flip spectrum vertically (only for better visialization, low freq. at bottom)
    private fun flip_vertically(mfccValues: Array<FloatArray>): Array<FloatArray> {
        val flip_mfccValues = Array(mfccValues.size) {
            FloatArray(mfccValues[0].size)
        }

        var k = mfccValues.size - 1
        for (i in mfccValues.indices) {
            for (s in mfccValues[0].indices) {
                flip_mfccValues[k][s] = mfccValues[i][s]
            }
            k -= 1
        }

        return flip_mfccValues
    }

    // Trim to desired shape if too large
    private fun reshape(mfccValues: Array<FloatArray>): Array<FloatArray> {

        val reshape = Array(128) {
            FloatArray(256)
        }

        for (i in reshape.indices) {
            for (s in reshape[0].indices) {
                reshape[i][s] = mfccValues[i][s]
            }
        }
        return reshape
    }

    private fun mode(arr: Array<IntArray>): IntArray {
        val oneDArray = IntArray(arr[0].size * arr.size)
        for (i in arr.indices) {
            for (s in arr[0].indices) {
                oneDArray[i * arr[i].size + s] = arr[i][s]
            }
        }
        return oneDArray
    }

    // FIR (Finite Impulse Response)

    private val ALPHA = 0.95F  // 0.95 ~ 0.97
    // FIR applies impulseResponse[i] to x[n - i], so this is y[n] = x[n] - 0.95 x[n-1] as in training
    private val PRE_EMPHASIS_IMPULSE_RESPONSE = floatArrayOf(1.0F, -ALPHA)

    // A fresh filter per window: training applies pre-emphasis to each 1 s clip on its own
    fun applyPreEmphasis(buf: FloatArray) {
        FIR(PRE_EMPHASIS_IMPULSE_RESPONSE).transform(buf)
    }

    private fun convertFromShortArrayToFloatArray(shortData: ShortArray): FloatArray {
        val size = shortData.size
        val floatData = FloatArray(size)
        for (i in 0 until size) {
            floatData[i] = shortData[i].toFloat()
        }
        return floatData
    }


    private fun convertFrom2DFloatArrayTo2DDoubleArray(floatData: Array<FloatArray>): Array<DoubleArray> {
        val doubleData = Array(floatData.size) {
            DoubleArray(floatData[0].size)
        }

        for (i in floatData.indices) {
            for (s in floatData[0].indices) {
                doubleData[i][s] = floatData[i][s].toDouble()
            }
        }

        return doubleData
    }

    private fun convertFrom2DDoubleArrayTo2DFloatArray(doubleData: Array<DoubleArray>): Array<FloatArray> {
        val floatData = Array(doubleData.size) {
            FloatArray(doubleData[0].size)
        }

        for (i in doubleData.indices) {
            for (s in doubleData[0].indices) {
                floatData[i][s] = doubleData[i][s].toFloat()
            }
        }

        return floatData
    }

    private fun stopRecord() {
        Log.v("stop record", "stop record")
        // The recording thread stops and releases its recorder on its next read
        recordingThread = null
        mAudioRecord = null
    }
}
