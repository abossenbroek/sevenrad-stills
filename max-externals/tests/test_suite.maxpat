{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 0,
			"architecture" : "x64",
			"modernui" : 1
		},
		"classnamespace" : "box",
		"rect" : [ 50.0, 50.0, 1400.0, 900.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "Automated test suite for SevenRad Max effects",
		"digest" : "Run all effect tests and save outputs for validation",
		"tags" : "jitter, GPU, test, automation, validation",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 20.0, 700.0, 24.0 ],
					"text" : "SevenRad Effects Test Suite - Automated Testing Framework",
					"fontsize" : 16.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 50.0, 800.0, 74.0 ],
					"text" : "This patcher tests all SevenRad effects against reference outputs.\n\nWORKFLOW:\n1. Load test image (input/test_image.png)\n2. Select effect and test case from dropdowns\n3. Click 'Run Test' to apply effect with parameters from reference/*.json\n4. Output saved to actual/*.png\n5. Use compare_outputs.py to validate PSNR/SSIM scores"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 150.0, 150.0, 20.0 ],
					"text" : "1. Load test image:",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 30.0, 180.0, 40.0, 40.0 ],
					"bgcolor" : [ 0.2, 0.6, 1.0, 1.0 ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.0, 230.0, 180.0, 22.0 ],
					"text" : "read input/test_image.png"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 30.0, 260.0, 139.0, 22.0 ],
					"text" : "jit.matrix test_input 4 char"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "jit.pwindow",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 30.0, 300.0, 256.0, 192.0 ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 320.0, 150.0, 200.0, 20.0 ],
					"text" : "2. Select effect to test:",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-21",
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"patching_rect" : [ 320.0, 180.0, 200.0, 22.0 ],
					"items" : [ "noise", ",", "saturation", ",", "chromatic", ",", "blur", ",", "blur_circular", ",", "motion", ",", "corduroy", ",", "bayer", ",", "bandswap", ",", "downscale", ",", "slcoff" ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 540.0, 150.0, 200.0, 20.0 ],
					"text" : "3. Select test case:",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "umenu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "", "" ],
					"patching_rect" : [ 540.0, 180.0, 250.0, 22.0 ],
					"items" : [ "0 - Light Gaussian noise", ",", "1 - Heavy Gaussian noise", ",", "2 - Horizontal scanlines", ",", "3 - Vertical artifacts" ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 820.0, 150.0, 150.0, 20.0 ],
					"text" : "4. Run test:",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-31",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 820.0, 180.0, 60.0, 60.0 ],
					"bgcolor" : [ 0.0, 0.8, 0.0, 1.0 ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-32",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 890.0, 195.0, 80.0, 20.0 ],
					"text" : "Run Test",
					"fontsize" : 14.0,
					"fontface" : 1
				}
			},
 			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 320.0, 250.0, 200.0, 20.0 ],
					"text" : "Test parameters (auto-loaded):"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-41",
					"maxclass" : "textedit",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "int", "", "" ],
					"patching_rect" : [ 320.0, 280.0, 300.0, 120.0 ],
					"text" : "Select an effect and test case to load parameters..."
				}
			},
 			{
				"box" : 				{
					"id" : "obj-50",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 650.0, 250.0, 200.0, 20.0 ],
					"text" : "Output preview:"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-51",
					"maxclass" : "jit.pwindow",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 650.0, 280.0, 320.0, 240.0 ]
				}
			},
 			{
				"box" : 				{
					"id" : "obj-60",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 520.0, 150.0, 20.0 ],
					"text" : "Test execution log:"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-61",
					"maxclass" : "textedit",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "", "int", "", "" ],
					"patching_rect" : [ 30.0, 550.0, 590.0, 150.0 ],
					"text" : "Ready to run tests...\n\nTest results will appear here."
				}
			},
 			{
				"box" : 				{
					"id" : "obj-70",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 650.0, 550.0, 320.0, 150.0 ],
					"text" : "EFFECT PROCESSING:\n\nSingle-pass effects:\n- sr.noise, sr.saturation, sr.chromatic\n- sr.blur.circular, sr.motion, sr.corduroy\n- sr.downscale\n\nTwo-pass effects:\n- sr.blur (h + v passes)\n- sr.bayer (mosaic + demosaic)\n\nHybrid CPU+GPU:\n- sr.bandswap (requires sr.tilegen)\n- sr.slcoff (requires sr.maskgen)"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-100",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 320.0, 220.0, 100.0, 22.0 ],
					"text" : "prepend effect"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-101",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 540.0, 220.0, 100.0, 22.0 ],
					"text" : "prepend case"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-102",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "" ],
					"patching_rect" : [ 320.0, 430.0, 200.0, 22.0 ],
					"text" : "route effect case"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-110",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 730.0, 150.0, 22.0 ],
					"text" : "print TEST_LOG"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-120",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1020.0, 20.0, 350.0, 200.0 ],
					"text" : "MANIFEST.JSON TEST CASES:\n\nnoise: 4 cases\nsaturation: 4 cases\nchromatic: 4 cases\nblur: 4 cases\nblur_circular: 3 cases\nmotion: 4 cases\ncorduroy: 3 cases\nbayer: 4 cases\nbandswap: 3 cases\ndownscale: 3 cases\nslcoff: 2 cases\n\nTotal: 38 test cases\n\nEach test loads parameters from:\nreference/{effect}_{case_id}.json\n\nAnd saves output to:\nactual/{effect}_{case_id}.png"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-130",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1020.0, 240.0, 350.0, 280.0 ],
					"text" : "VALIDATION WORKFLOW:\n\n1. Run all tests in this patcher\n2. Check actual/ folder for outputs\n3. Run validation script:\n   python compare_outputs.py\n\n4. Expected results:\n   - PSNR > 40 dB (PASS)\n   - SSIM > 0.99 (PASS)\n   - Max pixel diff ≤ 2 (PASS)\n\n5. If tests fail:\n   - Check effect implementation\n   - Verify parameter mappings\n   - Ensure RNG constants match\n   - Review shader GenExpr code\n\nKNOWN LIMITATIONS:\n- saltpepper: No test cases yet\n- corruption: No test cases yet\n- Hybrid effects require CPU externals\n\nRUN ALL TESTS:\nUse the Python script:\n  python tests/run_all_tests.py\n\nOr manually iterate through each\neffect + case combination in this UI."
				}
			},
 			{
				"box" : 				{
					"id" : "obj-200",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 820.0, 260.0, 100.0, 22.0 ],
					"text" : "jit.matrix test_out"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-201",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 820.0, 300.0, 150.0, 22.0 ],
					"text" : "write actual/output.png"
				}
			},
 			{
				"box" : 				{
					"id" : "obj-210",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 760.0, 700.0, 100.0 ],
					"text" : "NOTE: This test suite provides the UI framework for testing.\n\nFor GPU shader effects, you need to:\n1. Have a valid OpenGL context (jit.gl.render)\n2. Process images through jit.gl.pix with @gen pointing to .genjit files\n3. Use jit.gl.texture for GPU-side processing\n\nThe actual effect testing logic should be implemented in the subpatcher (p test_processor)\nor via an external Python/Node.js script that automates the Max application.\n\nFor now, use this patcher to manually test each effect one by one."
				}
			}
		],
		"lines" : [ 			{
				"patchline" : 				{
					"source" : [ "obj-11", 0 ],
					"destination" : [ "obj-12", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-12", 0 ],
					"destination" : [ "obj-13", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-13", 0 ],
					"destination" : [ "obj-14", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-21", 1 ],
					"destination" : [ "obj-100", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-23", 0 ],
					"destination" : [ "obj-101", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-100", 0 ],
					"destination" : [ "obj-102", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-101", 0 ],
					"destination" : [ "obj-102", 1 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-31", 0 ],
					"destination" : [ "obj-102", 2 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-102", 0 ],
					"destination" : [ "obj-41", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-102", 0 ],
					"destination" : [ "obj-61", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-102", 0 ],
					"destination" : [ "obj-110", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-200", 0 ],
					"destination" : [ "obj-51", 0 ]
				}
			},
 			{
				"patchline" : 				{
					"source" : [ "obj-201", 0 ],
					"destination" : [ "obj-200", 0 ]
				}
			}
		]
	}
}
