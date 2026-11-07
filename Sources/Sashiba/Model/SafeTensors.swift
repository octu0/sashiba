import Foundation

/// Minimal reader and writer of the safetensors format for float32 tensors: an 8-byte little-endian header length,
/// a JSON header naming each tensor's dtype, shape and byte range (plus a `__metadata__` map of strings), then the
/// raw little-endian data. The header is padded with spaces so the data starts on an 8-byte boundary.
public enum SafeTensors {
    public struct Tensor: Equatable, Sendable {
        public let shape: [Int]
        public let values: [Float]

        public init(shape: [Int], values: [Float]) {
            self.shape = shape
            self.values = values
        }
    }

    public enum Failure: Error {
        case truncated
        case badHeader
        case unsupportedType(String)
        case badShape(String)
    }

    /// Writes `tensors` in the given order.
    public static func write(_ tensors: [(name: String, tensor: Tensor)], metadata: [String: String], to url: URL) throws {
        var header: [String: Any] = [:]
        if metadata.isEmpty != true {
            header["__metadata__"] = metadata
        }
        var offset = 0
        for (name, tensor) in tensors {
            let count = tensor.shape.reduce(1, *)
            if count != tensor.values.count {
                throw Failure.badShape(name)
            }
            header[name] = ["dtype": "F32", "shape": tensor.shape, "data_offsets": [offset, offset + 4 * count]] as [String: Any]
            offset += 4 * count
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + json.count) % 8 != 0 {
            json.append(0x20)
        }
        var out = Data(capacity: 8 + json.count + offset)
        var length = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(json)
        for (_, tensor) in tensors {
            tensor.values.withUnsafeBufferPointer { out.append(UnsafeBufferPointer(start: $0.baseAddress, count: $0.count)) }
        }
        try out.write(to: url, options: .atomic)
    }

    /// Reads every tensor and the metadata.
    public static func read(from url: URL) throws -> (tensors: [String: Tensor], metadata: [String: String]) {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        if data.count < 8 {
            throw Failure.truncated
        }
        let length = data.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        if data.count < 8 + length {
            throw Failure.truncated
        }
        guard let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + length))) as? [String: Any] else {
            throw Failure.badHeader
        }
        var metadata: [String: String] = [:]
        var tensors: [String: Tensor] = [:]
        for (name, value) in header {
            if name == "__metadata__" {
                metadata = value as? [String: String] ?? [:]
                continue
            }
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else {
                throw Failure.badHeader
            }
            if dtype != "F32" {
                throw Failure.unsupportedType(dtype)
            }
            let start = 8 + length + offsets[0]
            let end = 8 + length + offsets[1]
            let count = shape.reduce(1, *)
            if data.count < end || end - start != 4 * count {
                throw Failure.badShape(name)
            }
            let values = [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                if count == 0 {
                    initialized = 0
                    return
                }
                data.withUnsafeBytes { raw in
                    _ = memcpy(buffer.baseAddress!, raw.baseAddress! + start, 4 * count)
                }
                initialized = count
            }
            tensors[name] = Tensor(shape: shape, values: values)
        }
        return (tensors: tensors, metadata: metadata)
    }
}

