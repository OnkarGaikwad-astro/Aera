import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'tokenizer.dart';

class EmbeddingService {
  late Interpreter _interpreter;
  late Tokenizer _tokenizer;
  bool _isInitialized = false;


  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/embedding_model/model.tflite',
    );

    _tokenizer = await Tokenizer.load(
      'assets/embedding_model/vocab.txt',
    );
    _isInitialized = true;
  }

  List<double> generateEmbedding(String text) {
    if (!_isInitialized) {
      throw Exception("EmbeddingService not initialized");
    }
    text = text.toLowerCase().trim();
    final tokens = _tokenizer.tokenize(text);

    final inputIds = List.filled(128, 0);
    final attentionMask = List.filled(128, 0);

    for (int i = 0; i < tokens.length && i < 128; i++) {
      inputIds[i] = tokens[i];
      attentionMask[i] = 1;
    }

    final output = List.generate(
      1,
      (_) => List.filled(384, 0.0),
    );

    _interpreter.runForMultipleInputs(
      [
        [inputIds],
        [attentionMask],
      ],
      {
    0: output, 
  },
    );

    return output[0];
  }
}