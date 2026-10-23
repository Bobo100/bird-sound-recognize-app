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
    private var isRecording = false

    private var laiyuan = MediaRecorder.AudioSource.VOICE_RECOGNITION //来源
    private val SAMPLE_RATE = 44100 //采样频率
    private val SAMPLE_DURATION_MS = 1000
    private val CHANNEL = AudioFormat.CHANNEL_IN_MONO//声道
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT //格式
    private val MONO_RECORDING_LENGTH = (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000)

    var bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, audioFormat) //缓冲区大小

    private lateinit var mAcousticEventDetector: AcousticEventDetector

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_real_time_recognize3)

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

        // Ask for the microphone up front; startRecord() does nothing without it
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
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

    private fun startRecord() {
        Log.v("start record", "start record")

        val audioBuffer = ShortArray(bufferSize)
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
            return
        }
        mAudioRecord = AudioRecord(
            laiyuan, SAMPLE_RATE,
            CHANNEL, audioFormat, bufferSize
        )
        mAudioRecord!!.startRecording()
        isRecording = true
        val shortInputBuffertemp = ShortArray(MONO_RECORDING_LENGTH)

        Thread {
            var loop_check = false

            var count = 0
            while (isRecording) {
                val read = mAudioRecord!!.read(audioBuffer, 0, audioBuffer.size) //3528
                //有沒有正確讀取
                if (AudioRecord.ERROR_INVALID_OPERATION != read) {
                    // Ring buffer
                    //代表是最後一次
                    if (shortInputBuffertemp.size / audioBuffer.size == count) {
                        loop_check = true

                        System.arraycopy(
                            audioBuffer,
                            0,
                            shortInputBuffertemp,
                            count * audioBuffer.size,
                            shortInputBuffertemp.size - (count * audioBuffer.size)
                        )
                    } else {
                        System.arraycopy(
                            audioBuffer,
                            0,
                            shortInputBuffertemp,
                            count * audioBuffer.size,
                            audioBuffer.size
                        )
                    }

                    ++count

                    if (loop_check) {
                        val temp = convertFromShortArrayToFloatArray(shortInputBuffertemp)
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
                        val results = mAcousticEventDetector.recognize(oneDArray)
                        // Views may only be touched on the UI thread
                        runOnUiThread {
                            findViewById<TextView>(R.id.recognize_result).text = "$results"
                        }
                        Log.v("results", results.toString())
                        ///////////////////////////////////////////////////////////////////////////////////////////

                        loop_check = false
                        count = 0
                    }
                }
            }
        }.start()
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
    private val PRE_EMPHASIS_MPULSE_RESPONSE = floatArrayOf(-ALPHA, 1.0F)

    private var mPreEmphasisFir = FIR(PRE_EMPHASIS_MPULSE_RESPONSE) //0.95

    fun applyPreEmphasis(buf: FloatArray) {
        mPreEmphasisFir.transform(buf)
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
        isRecording = false
        if (mAudioRecord != null) {
            mAudioRecord!!.stop()
            mAudioRecord!!.release()
            mAudioRecord = null
        }
    }
}